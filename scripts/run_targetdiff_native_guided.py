#!/usr/bin/env python3
"""TargetDiff + ESField — monkey-patches native sample_diffusion for guidance.

Strategy: Use TargetDiff's native `model.sample_diffusion()` which is verified to work.
We monkey-patch the method to add post-step guidance callbacks.

Modes: unguided (native), hard_fix (overwrite anchors), kinematic (CoM attraction)
"""

import argparse, json, sys, time
import types as _types  # renamed to avoid conflict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm import tqdm
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
TD = Path("/root/baselines/TargetDiff/code/targetdiff-main")

# ESField first
sys.path.insert(0, str(ROOT / "src"))
from guidance.latent_guidance import SiteCompatibilityEnergy
from guidance.kinematic_anchor import KinematicScheduler

# TargetDiff
sys.path.insert(0, str(TD))
for _k in list(sys.modules):
    if _k.startswith(('models', 'utils', 'datasets')):
        del sys.modules[_k]

import utils.misc as misc
import utils.transforms as trans
from utils import reconstruct
from utils.evaluation import atom_num
from models.molopt_score_model import ScorePosNet3D
from models.molopt_score_model import extract as _extract
from models.molopt_score_model import index_to_log_onehot, log_sample_categorical
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData, torchify_dict
from utils.data import PDBProtein

TD_CKPT = "/root/autodl-tmp/checkpoints/TargetDiff/pretrained_diffusion.pt"
SITE_MAP_DIR = ROOT / "experiments/targetdiff_replication/site_maps"
POCKET_CFG = {
    "3mfw": {"year": "2001-2010"},
    "6o4x": {"year": "2011-2019"},
    "2gni": {"year": "2001-2010"},
}


def load_model(device="cuda:0"):
    ckpt = torch.load(TD_CKPT, map_location=device, weights_only=False)
    config = ckpt["config"]
    pf = trans.FeaturizeProteinAtom()
    lf = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    model = ScorePosNet3D(config.model, pf.feature_dim, lf.feature_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, pf


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0], dtype=torch.long),
        }
    )
    return Compose([trans.FeaturizeProteinAtom()])(data)


# ═══════════════════════════════════════════════════════════
# Monkey-patch sample_diffusion for guidance
# ═══════════════════════════════════════════════════════════

def make_guided_sample_diffusion(mode, anchor_indices, anchor_targets, site_energy, lambda_max):
    """Return a patched version of model.sample_diffusion with guidance injection."""

    def guided_sample_diffusion(self, protein_pos, protein_v, batch_protein,
                                 init_ligand_pos, init_ligand_v, batch_ligand,
                                 num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        # Use the imported center_pos function from molopt_score_model
        from models.molopt_score_model import center_pos as center_pos_fn
        protein_pos, init_ligand_pos, offset = center_pos_fn(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        kin_scheduler = None
        if mode == "kinematic" and site_energy is not None:
            kin_scheduler = KinematicScheduler(lambda_max=lambda_max, profile="quadratic")

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for step_i, i_val in enumerate(time_seq):
            t = torch.full(size=(num_graphs,), fill_value=i_val, dtype=torch.long, device=protein_pos.device)
            with torch.no_grad():
                preds = self(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )

                if self.model_mean_type == 'noise':
                    pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                    pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                    v0_from_e = preds['pred_ligand_v']
                elif self.model_mean_type == 'C0':
                    pos0_from_e = preds['pred_ligand_pos']
                    v0_from_e = preds['pred_ligand_v']
                else:
                    raise ValueError

                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = _extract(self.posterior_logvar, t, batch_ligand)
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)

                # ── GUIDANCE INJECTION ──
                if mode == "hard_fix" and anchor_targets is not None:
                    for mol_i in range(num_graphs):
                        mi = (batch_ligand == mol_i)
                        mol_idxs = torch.where(mi)[0]
                        na = len(mol_idxs)
                        for ai in anchor_indices:
                            if 0 <= ai < na and ai < len(anchor_targets):
                                ligand_pos_next[mol_idxs[ai]] = anchor_targets[ai].to(ligand_pos_next.device)

                elif mode == "kinematic" and site_energy is not None and kin_scheduler is not None:
                    t_norm = step_i / max(len(time_seq) - 1, 1)
                    lam = kin_scheduler(t_norm)
                    if isinstance(lam, torch.Tensor):
                        lam = lam.item()

                    if lam > 0 and site_energy.n_sites > 0 and anchor_indices:
                        sc = site_energy._site_centers.to(ligand_pos_next.device)
                        sigma2 = 2.0 * 3.0 ** 2
                        cmat = site_energy.compatibility_matrix.to(ligand_pos_next.device)
                        eidx = site_energy._site_env_indices.to(ligand_pos_next.device)
                        best = cmat[eidx].max(dim=-1).values

                        for mol_i in range(num_graphs):
                            mi = (batch_ligand == mol_i)
                            mol_idxs = torch.where(mi)[0]
                            na = len(mol_idxs)
                            valid_a = [ai for ai in anchor_indices if 0 <= ai < na]
                            if not valid_a:
                                continue

                            apos = ligand_pos_next[mol_idxs[valid_a]]
                            acom = apos.mean(dim=0)

                            rel = sc - acom.unsqueeze(0)
                            dsq = (rel ** 2).sum(dim=-1)
                            gauss = torch.exp(-dsq / sigma2)
                            w = gauss * best
                            if site_energy._site_confs is not None:
                                w = w * site_energy._site_confs.to(ligand_pos_next.device)

                            grad = (w.unsqueeze(-1) * rel / sigma2).sum(dim=0)
                            gn = grad.norm()
                            if gn > 1e-8:
                                grad = grad * (0.05 / gn)

                            corr = lam * grad
                            cn = corr.norm()
                            if cn > 0.5:
                                corr = corr * (0.5 / cn)

                            ligand_pos_next[mol_idxs[valid_a]] = apos + corr.unsqueeze(0)
                # ── END GUIDANCE ──

                ligand_pos = ligand_pos_next

                if not pos_only:
                    log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                    log_ligand_v_cur = index_to_log_onehot(ligand_v, self.num_classes)
                    log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v_cur, t, batch_ligand)
                    ligand_v_next = log_sample_categorical(log_model_prob)
                    v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                    vt_pred_traj.append(log_model_prob.clone().cpu())
                    ligand_v = ligand_v_next

                ori_ligand_pos = ligand_pos + offset[batch_ligand]
                pos_traj.append(ori_ligand_pos.clone().cpu())
                v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos, 'v': ligand_v,
            'pos_traj': pos_traj, 'v_traj': v_traj,
            'v0_traj': v0_pred_traj, 'vt_traj': vt_pred_traj
        }

    return guided_sample_diffusion


# ═══════════════════════════════════════════════════════════
# Native-style sampling wrapper (from sample_diffusion_ligand)
# ═══════════════════════════════════════════════════════════

def sample_diffusion_ligand_wrapped(model, data, num_samples,
                                     batch_size=8, device='cuda:0',
                                     num_steps=None, pos_only=False,
                                     center_pos_mode='protein',
                                     sample_num_atoms='prior'):
    all_pred_pos, all_pred_v = [], []

    num_batch = int(np.ceil(num_samples / batch_size))
    for i in tqdm(range(num_batch), desc='  sampling'):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)],
                                      follow_batch=FOLLOW_BATCH).to(device)

        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            else:
                raise ValueError

            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v = r['pos'], r['v']

            # Unbatch
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_pred_pos.append(ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])

            ligand_v_array = ligand_v.cpu().numpy()
            for k in range(n_data):
                all_pred_v.append(ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])

    return all_pred_pos, all_pred_v


def reconstruct_save(all_pos, all_v, output_dir, prefix="mol"):
    sdf_dir = Path(output_dir) / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    valid = []
    for i, (pos, v) in enumerate(zip(all_pos, all_v)):
        try:
            atom_types = trans.get_atomic_number_from_index(v, mode='add_aromatic')
            aromatic = trans.is_aromatic_from_index(v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pos, atom_types, aromatic)
            smi = Chem.MolToSmiles(mol)
            if '.' not in smi:
                mol.SetProp("_Name", f"{prefix}_{i:03d}")
                Chem.MolToMolFile(mol, str(sdf_dir / f"{prefix}_{i:03d}.sdf"))
                valid.append({"mol": mol, "idx": i, "smiles": smi, "pos": pos, "v": v})
        except Exception:
            pass
    return valid


def compute_direct_occ(mols_data, site_map, threshold=2.5):
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return 0.0
    occupied = 0
    for md in mols_data:
        pos = md["pos"]
        for site in hew_sites:
            sc = np.array(site["center"])
            if np.linalg.norm(pos - sc, axis=-1).min() <= threshold:
                occupied += 1
                break
    return occupied / max(len(mols_data), 1)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", required=True, choices=["3mfw", "6o4x", "2gni"])
    parser.add_argument("--mode", default="unguided",
                        choices=["unguided", "hard_fix", "kinematic", "all"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="experiments/targetdiff_native_guided")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    pocket = args.pocket
    year = POCKET_CFG[pocket]["year"]
    outdir = Path(args.output_dir) / pocket
    outdir.mkdir(parents=True, exist_ok=True)

    # Site map
    with open(SITE_MAP_DIR / f"{pocket}_site_map.json") as f:
        site_map = json.load(f)
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    print(f"{pocket}: {len(hew_sites)} HEW")

    # Site energy
    se = SiteCompatibilityEnergy(sigma_distance=3.0)
    if hew_sites:
        sc = torch.tensor([s["center"] for s in hew_sites], dtype=torch.float32)
        se.register_sites(sc, torch.zeros(len(hew_sites), dtype=torch.long))
    se.to(args.device)

    # Model
    model, cfg, _ = load_model(args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Protein
    pdb_path = f"/root/autodl-tmp/data/PDB/P-L/{year}/{pocket}/{pocket}_pocket.pdb"
    data = pdb_to_pocket_data(pdb_path)
    print(f"Pocket: {len(data.protein_pos)} atoms")

    # Anchor config
    anchor_indices = [0, 1, 2, 3]
    if hew_sites:
        best_hew = sorted(hew_sites, key=lambda s: s.get("confidence", 0), reverse=True)[0]
        anchor_targets = torch.tensor([best_hew["center"]] * 4, dtype=torch.float32)
    else:
        anchor_targets = torch.zeros(4, 3)

    modes = ["unguided", "hard_fix", "kinematic"] if args.mode == "all" else [args.mode]

    # Save original sample_diffusion
    original_sample_diffusion = model.sample_diffusion

    summaries = {}
    for mode in modes:
        print(f"\n{'='*50}\n[{pocket}] {mode} ({args.n_samples} molecules, {args.num_steps} steps)\n{'='*50}")

        # Apply monkey-patch
        at = anchor_targets if mode == "hard_fix" else None
        site_e = se if mode == "kinematic" else None
        patched_fn = make_guided_sample_diffusion(mode, anchor_indices, at, site_e, args.lambda_max)
        model.sample_diffusion = _types.MethodType(patched_fn, model)

        t0 = time.time()
        positions, types = sample_diffusion_ligand_wrapped(
            model, data, args.n_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=args.num_steps,
        )
        elapsed = time.time() - t0

        # Restore original
        model.sample_diffusion = original_sample_diffusion

        # Reconstruct
        mode_dir = outdir / mode
        valid = reconstruct_save(positions, types, mode_dir, prefix=mode)
        direct_occ = compute_direct_occ(
            [{"pos": p, "v": v} for p, v in zip(positions, types)], site_map
        )

        print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(positions),1):.1f}s/mol)")
        print(f"  Valid: {len(valid)}/{len(positions)}")
        print(f"  DirectOcc: {direct_occ:.1%}")

        if valid:
            print(f"  Sample SMILES: {[v['smiles'] for v in valid[:3]]}")

        torch.save({
            "positions": positions, "types": types,
            "valid": len(valid), "direct_occ": direct_occ,
        }, mode_dir / "results.pt")

        summaries[mode] = {
            "direct_occ": direct_occ,
            "n_valid": len(valid),
            "n_total": len(positions),
            "time": elapsed,
        }

    # Summary
    print(f"\n{'='*50}\nRESULTS: {pocket}\n{'='*50}")
    print(f"{'Condition':<15} {'DirectOcc':>10} {'Valid':>10}")
    print("-" * 35)
    for mode in modes:
        s = summaries[mode]
        print(f"{mode:<15} {s['direct_occ']:>9.1%} {s['n_valid']:>5}/{s['n_total']:<5}")

    print("\nLaTeX table rows:")
    for mode in modes:
        s = summaries[mode]
        print(f"  {pocket} & {mode.capitalize()} & {s['direct_occ']:.1%} & "
              f"TBD & TBD & TBD & {s['n_valid']}/{s['n_total']} \\\\")

    with open(outdir / "summary.json", "w") as f:
        json.dump({"pocket": pocket, "generator": "TargetDiff",
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "conditions": summaries}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
