#!/usr/bin/env python3
"""TargetDiff + ESField v2 — Uses native TargetDiff sampling with post-step guidance.

Key difference from v1: Uses model.sample_diffusion() directly rather than
re-implementing the DDPM loop. This guarantees native-quality reconstruction.

Modes:
  unguided  — native TargetDiff sampling
  hard_fix  — overlay anchor atoms to HEW site centers after each step
  kinematic — CoM-only attraction toward HEW sites (kinematic anchor guidance)
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TD = Path("/root/baselines/TargetDiff/code/targetdiff-main")

# Import ESField BEFORE TargetDiff (path order matters)
sys.path.insert(0, str(ROOT / "src"))
from guidance.latent_guidance import SiteCompatibilityEnergy
from guidance.kinematic_anchor import KinematicScheduler

# Now TargetDiff
sys.path.insert(0, str(TD))
# Clear cached modules that conflict
for _k in list(sys.modules):
    if _k.startswith(('models', 'utils', 'datasets')):
        del sys.modules[_k]

from models.molopt_score_model import ScorePosNet3D, extract as _td_extract, index_to_log_onehot, log_sample_categorical
from utils import transforms as trans, reconstruct
from utils.evaluation import atom_num
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData, torchify_dict
from utils.data import PDBProtein
from rdkit import Chem

# ── Config ──
TD_CKPT = "/root/autodl-tmp/checkpoints/TargetDiff/pretrained_diffusion.pt"
POCKET_CFG = {
    "3mfw": {"year": "2001-2010", "hew": 7},
    "6o4x": {"year": "2011-2019", "hew": 6},
    "2gni": {"year": "2001-2010", "hew": 3},
}
SITE_MAP_DIR = ROOT / "experiments/targetdiff_replication/site_maps"


def load_model(device="cuda:0"):
    ckpt = torch.load(TD_CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    pf = trans.FeaturizeProteinAtom()
    lf = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    model = ScorePosNet3D(cfg.model, pf.feature_dim, lf.feature_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, pf


def load_pocket(pdb_path, pf):
    pocket = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket),
        ligand_dict={
            'element': torch.empty([0], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0], dtype=torch.long),
        }
    )
    return Compose([pf])(data)


def center_pos_fn(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        return protein_pos, ligand_pos, torch.zeros(len(ligand_pos), 3).to(protein_pos)
    offset = scatter_mean(protein_pos, batch_protein, dim=0)[batch_ligand]
    return protein_pos, ligand_pos - offset, offset


# ═══════════════════════════════════════════════════════════════════
# Wrapped sampling with post-step guidance
# ═══════════════════════════════════════════════════════════════════

def sample_diffusion_wrapped(
    model, config, protein_data,
    n_samples=50, device="cuda:0",
    mode="unguided",
    anchor_indices=None,
    anchor_targets=None,  # for hard_fix: Tensor[n_anchors, 3]
    site_energy=None,      # for kinematic
    lambda_max=1.0,
    num_steps=None,
):
    """Native TargetDiff sampling with post-step guidance injection."""
    num_diff_steps = num_steps or config.sample.num_steps
    batch_size = 8

    all_pos, all_v = [], []
    bsz = batch_size
    n_batches = int(np.ceil(n_samples / bsz))

    kin_scheduler = None
    if mode == "kinematic" and site_energy is not None:
        kin_scheduler = KinematicScheduler(lambda_max=lambda_max, profile="quadratic")

    for batch_i in tqdm(range(n_batches), desc=f'  {mode}'):
        n_curr = min(bsz, n_samples - batch_i * bsz)
        batch = Batch.from_data_list(
            [protein_data.clone() for _ in range(n_curr)],
            follow_batch=FOLLOW_BATCH
        ).to(device)
        bp = batch.protein_element_batch

        # Atom count sampling
        pocket_size = atom_num.get_space_size(protein_data.protein_pos.numpy())
        n_atoms_list = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_curr)]
        bl = torch.repeat_interleave(torch.arange(n_curr), torch.tensor(n_atoms_list)).to(device)

        # Init
        center_pts = scatter_mean(batch.protein_pos, bp, dim=0)
        bc = center_pts[bl]
        pos = bc + torch.randn_like(bc) * 2.0
        uniform_logits = torch.zeros(len(bl), model.num_classes, device=device)
        v = log_sample_categorical(uniform_logits)

        _, pos, offset = center_pos_fn(batch.protein_pos, pos, bp, bl, mode="protein")

        # DDPM loop
        time_seq = list(reversed(range(model.num_timesteps - num_diff_steps, model.num_timesteps)))
        for step_i, t_val in enumerate(time_seq):
            t = torch.full(size=(n_curr,), fill_value=t_val, dtype=torch.long, device=device)

            # Forward pass (CRITICAL: torch.no_grad() prevents OOM)
            with torch.no_grad():
                preds = model(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=bp,
                    init_ligand_pos=pos,
                    init_ligand_v=v,
                    batch_ligand=bl,
                    time_step=t
                )

                # x0 prediction
                if model.model_mean_type == 'C0':
                    pos0 = preds['pred_ligand_pos']
                    v0 = preds['pred_ligand_v']
                else:
                    pos0 = model._predict_x0_from_eps(xt=pos, eps=preds['pred_ligand_pos'] - pos, t=t, batch=bl)
                    v0 = preds['pred_ligand_v']

                # DDPM posterior
                pos_mean = model.q_pos_posterior(x0=pos0, xt=pos, t=t, batch=bl)
                pos_logvar = _td_extract(model.posterior_logvar, t, bl)
                nonzero_mask = (1 - (t == 0).float())[bl].unsqueeze(-1)
                pos_next = pos_mean + nonzero_mask * (0.5 * pos_logvar).exp() * torch.randn_like(pos)

                # ── GUIDANCE INJECTION ──
                if mode == "hard_fix" and anchor_targets is not None:
                    for mol_i in range(n_curr):
                        mi = (bl == mol_i)
                        mol_idxs = torch.where(mi)[0]
                        na = len(mol_idxs)
                        for ai in anchor_indices:
                            if 0 <= ai < na and ai < len(anchor_targets):
                                pos_next[mol_idxs[ai]] = anchor_targets[ai].to(device)

                elif mode == "kinematic" and site_energy is not None and kin_scheduler is not None and anchor_indices:
                    t_norm = step_i / max(len(time_seq) - 1, 1)
                    lam = kin_scheduler(t_norm)
                    if isinstance(lam, torch.Tensor):
                        lam = lam.item()

                    if lam > 0 and site_energy.n_sites > 0:
                        for mol_i in range(n_curr):
                            mi = (bl == mol_i)
                            mol_idxs = torch.where(mi)[0]
                            na = len(mol_idxs)
                            valid_a = [ai for ai in anchor_indices if 0 <= ai < na]
                            if not valid_a:
                                continue

                            apos = pos_next[mol_idxs[valid_a]]
                            acom = apos.mean(dim=0)

                            # Site gradient at anchor CoM
                            sc = site_energy._site_centers.to(device)
                            sigma2 = 2.0 * 3.0 ** 2
                            rel = sc - acom.unsqueeze(0)
                            dsq = (rel ** 2).sum(dim=-1)
                            gauss = torch.exp(-dsq / sigma2)

                            cmat = site_energy.compatibility_matrix.to(device)
                            eidx = site_energy._site_env_indices.to(device)
                            best = cmat[eidx].max(dim=-1).values

                            w = gauss * best
                            if site_energy._site_confs is not None:
                                w = w * site_energy._site_confs.to(device)

                            grad = (w.unsqueeze(-1) * rel / sigma2).sum(dim=0)
                            gn = grad.norm()
                            if gn > 1e-8:
                                grad = grad * (0.05 / gn)

                            corr = lam * grad
                            cn = corr.norm()
                            if cn > 0.5:
                                corr = corr * (0.5 / cn)

                            # Pure translation
                            pos_next[mol_idxs[valid_a]] = apos + corr.unsqueeze(0)

                pos = pos_next

                # Atom type update
                if v0 is not None:
                    log_v_recon = F.log_softmax(v0, dim=-1)
                    log_v = index_to_log_onehot(v, model.num_classes)
                    log_prob = model.q_v_posterior(log_v_recon, log_v, t, bl)
                    v = log_sample_categorical(log_prob)
            # end torch.no_grad()

        pos_final = pos + offset[bl]
        pos_np = pos_final.detach().cpu().numpy().astype(np.float64)
        v_np = v.detach().cpu().numpy()

        cum = np.cumsum([0] + n_atoms_list)
        for k in range(n_curr):
            all_pos.append(pos_np[cum[k]:cum[k+1]])
            all_v.append(v_np[cum[k]:cum[k+1]])

    return all_pos, all_v


# ═══════════════════════════════════════════════════════════════════
# Reconstruction (using TargetDiff native)
# ═══════════════════════════════════════════════════════════════════

def reconstruct_native(all_pos, all_v, output_dir, prefix="mol"):
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
    return occupied / len(mols_data) if mols_data else 0.0


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", required=True, choices=["3mfw", "6o4x", "2gni"])
    parser.add_argument("--mode", default="all",
                        choices=["unguided", "hard_fix", "kinematic", "all"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="experiments/targetdiff_v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pocket = args.pocket
    pcfg = POCKET_CFG[pocket]
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
    model, cfg, pf = load_model(args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Protein
    year = pcfg["year"]
    pdb_path = f"/root/autodl-tmp/data/PDB/P-L/{year}/{pocket}/{pocket}_pocket.pdb"
    pdata = load_pocket(pdb_path, pf)
    print(f"Pocket: {len(pdata.protein_pos)} atoms")

    # Anchor config
    anchor_indices = [0, 1, 2, 3]
    if hew_sites:
        best = sorted(hew_sites, key=lambda s: s.get("confidence", 0), reverse=True)[0]
        anchor_targets = torch.tensor([best["center"]] * 4, dtype=torch.float32)
    else:
        anchor_targets = torch.zeros(4, 3)

    n_each = 4 if args.dry_run else args.n_samples
    modes = ["unguided", "hard_fix", "kinematic"] if args.mode == "all" else [args.mode]

    summaries = {}
    for mode in modes:
        print(f"\n{'='*50}\n[{pocket}] {mode} ({n_each} molecules)\n{'='*50}")
        t0 = time.time()

        at = anchor_targets if mode == "hard_fix" else None
        site_e = se if mode == "kinematic" else None

        positions, types = sample_diffusion_wrapped(
            model, cfg, pdata,
            n_samples=n_each, device=args.device,
            mode=mode,
            anchor_indices=anchor_indices,
            anchor_targets=at,
            site_energy=site_e,
            lambda_max=args.lambda_max,
            num_steps=args.num_steps,
        )

        # Reconstruct
        valid = reconstruct_native(positions, types, str(outdir / mode), prefix=mode)
        direct_occ = compute_direct_occ(
            [{"pos": p, "v": v} for p, v in zip(positions, types)], site_map
        )

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(positions),1):.1f}s/mol)")
        print(f"  Valid: {len(valid)}/{len(positions)}")
        print(f"  DirectOcc: {direct_occ:.1%}")

        torch.save({
            "positions": positions, "types": types,
            "valid_count": len(valid), "direct_occ": direct_occ,
        }, outdir / mode / "results.pt")

        summaries[mode] = {
            "direct_occ": direct_occ,
            "n_valid": len(valid),
            "n_total": len(positions),
            "time": elapsed,
        }

        if valid:
            smi_sample = [v["smiles"] for v in valid[:5]]
            print(f"  Sample SMILES: {smi_sample}")

    # Summary
    print(f"\n{'='*50}\nRESULTS: {pocket}\n{'='*50}")
    print(f"{'Condition':<15} {'DirectOcc':>10} {'Valid':>10}")
    print("-" * 35)
    for mode in modes:
        s = summaries[mode]
        print(f"{mode:<15} {s['direct_occ']:>9.1%} {s['n_valid']:>5}/{s['n_total']:<5}")

    # LaTeX
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
