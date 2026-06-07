#!/usr/bin/env python3
"""TargetDiff + ESField Full Pipeline: Unguided / Hard-Fix / Kinematic Anchor Guidance.

Replicates core findings on TargetDiff (DDPM diffusion model) to demonstrate
model-agnosticity of kinematic anchor guidance.

Modes:
  - unguided:    Raw TargetDiff DDPM sampling (water-blind baseline)
  - hard_fix:    Hard-overwrite anchor atom coordinates to HEW sites after each step
  - kinematic:   CoM-only soft attraction toward HEW sites (kinematic anchor guidance)

Outputs:
  - SDF files for each generated molecule
  - DirectOcc (%) — fraction of molecules with >=1 atom within 2.5Å of any HEW site
  - KPE ratio — fraction of total kinetic energy from guidance
  - Per-condition summary JSON

Usage:
  # Single pocket, dry run
  python scripts/run_targetdiff_full_pipeline.py --pocket 3mfw --mode all --n-samples 10 --dry-run

  # Full run
  python scripts/run_targetdiff_full_pipeline.py --pocket 3mfw --mode all --n-samples 50
  python scripts/run_targetdiff_full_pipeline.py --pocket 6o4x --mode all --n-samples 50
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm import tqdm
from rdkit import Chem

# ── Path setup ──
ROOT = Path(__file__).resolve().parents[1]
TD = Path("/root/baselines/TargetDiff/code/targetdiff-main")

# IMPORTANT: Import ESField modules BEFORE adding TargetDiff to path.
# Both projects have a `models` package — TargetDiff's would shadow ESField's.
sys.path.insert(0, str(ROOT / "src"))
from guidance.latent_guidance import SiteCompatibilityEnergy
from guidance.kinematic_anchor import KinematicScheduler
# Save ESField module refs before TargetDiff pollutes the namespace
import guidance as _esfield_guidance

# Now add TargetDiff to path (its models/ will resolve to TargetDiff's)
sys.path.insert(0, str(TD))
# Clear any cached ESField module names that conflict with TargetDiff
for _k in list(sys.modules):
    if _k.startswith('models') or _k.startswith('utils') or _k.startswith('datasets'):
        del sys.modules[_k]

# ── TargetDiff imports ──
from models.molopt_score_model import ScorePosNet3D, extract, index_to_log_onehot, log_sample_categorical
from utils import transforms as trans
from utils import reconstruct
from utils.evaluation import atom_num
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData, torchify_dict
from utils.data import PDBProtein


# ═══════════════════════════════════════════════════════════════════════════════
# Pocket configuration
# ═══════════════════════════════════════════════════════════════════════════════

POCKET_CONFIG = {
    "3mfw": {"year": "2001-2010", "n_hew": 7, "ref_ligand_atoms": 26},
    "6o4x": {"year": "2011-2019", "n_hew": 6, "ref_ligand_atoms": 22},
    "2gni": {"year": "2001-2010", "n_hew": 3, "ref_ligand_atoms": 20},
}

TD_CKPT = "/root/autodl-tmp/checkpoints/TargetDiff/pretrained_diffusion.pt"
SITE_MAP_DIR = ROOT / "experiments/targetdiff_replication/site_maps"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_config(device="cuda:0"):
    ckpt = torch.load(TD_CKPT, map_location=device, weights_only=False)
    config = ckpt["config"]
    pf = trans.FeaturizeProteinAtom()
    ligand_mode = config.data.transform.ligand_atom_mode
    lf = trans.FeaturizeLigandAtom(ligand_mode)
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=pf.feature_dim,
        ligand_atom_feature_dim=lf.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, pf


def load_protein(pdb_path, protein_featurizer):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
        }
    )
    transform = Compose([protein_featurizer])
    return transform(data)


def center_pos_fn(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = torch.zeros(len(ligand_pos), 3).to(protein_pos)
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)[batch_ligand]
    else:
        raise NotImplementedError(mode)
    return protein_pos, ligand_pos - offset, offset


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Full Molecule Generation with Guidance Mode
# ═══════════════════════════════════════════════════════════════════════════════

def generate_molecules_targetdiff(
    model, config, protein_data,
    n_samples=50, device="cuda:0",
    mode="unguided",       # "unguided" | "hard_fix" | "kinematic"
    anchor_indices=None,   # list[int] — which atoms are anchors
    anchor_targets=None,   # Tensor[N_anchors, 3] — target coordinates for hard_fix
    site_energy=None,      # SiteCompatibilityEnergy for kinematic
    kinematic_lambda_max=1.0,
    num_steps=None,
    track_kpe=True,
    batch_size=10,
):
    num_diff_steps = num_steps or config.sample.num_steps
    n_batches = int(np.ceil(n_samples / batch_size))

    all_pos, all_v = [], []
    kpe_ode_total = 0.0
    kpe_guide_total = 0.0

    for batch_i in tqdm(range(n_batches), desc=f'  {mode}'):
        n_curr = min(batch_size, n_samples - batch_i * batch_size)
        batch = Batch.from_data_list(
            [protein_data.clone() for _ in range(n_curr)],
            follow_batch=FOLLOW_BATCH
        ).to(device)
        bp = batch.protein_element_batch

        # Sample atom counts from prior
        pocket_size = atom_num.get_space_size(protein_data.protein_pos.numpy())
        ligand_n_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_curr)]
        bl = torch.repeat_interleave(torch.arange(n_curr), torch.tensor(ligand_n_atoms)).to(device)

        # Initialize positions and types
        # Start from protein centre + small noise
        center_pts = scatter_mean(batch.protein_pos, bp, dim=0)
        bc = center_pts[bl]
        pos = bc + torch.randn_like(bc) * 3.0  # 3Å initial spread
        # Sample initial atom types from uniform logits (like TargetDiff sample_diffusion_ligand)
        uniform_logits = torch.zeros(len(bl), model.num_classes, device=device)
        v = log_sample_categorical(uniform_logits)

        _, pos, offset = center_pos_fn(
            batch.protein_pos, pos, bp, bl, mode="protein"
        )

        # Kinematic scheduler
        kin_scheduler = None
        if mode == "kinematic" and site_energy is not None:
            kin_scheduler = KinematicScheduler(
                lambda_max=kinematic_lambda_max, profile="quadratic"
            )

        # DDPM time steps (reverse: T → 0)
        time_seq = list(reversed(range(model.num_timesteps - num_diff_steps, model.num_timesteps)))

        for step_i, t_val in enumerate(time_seq):
            t = torch.full(size=(n_curr,), fill_value=t_val, dtype=torch.long, device=device)
            pos_before = pos.clone() if track_kpe else None

            with torch.no_grad():
                preds = model(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=bp,
                    init_ligand_pos=pos,
                    init_ligand_v=v,
                    batch_ligand=bl,
                    time_step=t,
                )

            # x0 prediction (mean_type = "C0" for TargetDiff checkpoint)
            if model.model_mean_type == 'C0':
                pos0 = preds['pred_ligand_pos']
                v0 = preds['pred_ligand_v']
            else:
                pos0 = model._predict_x0_from_eps(
                    xt=pos, eps=preds['pred_ligand_pos'] - pos, t=t, batch=bl
                )
                v0 = preds['pred_ligand_v']

            # DDPM posterior mean and noise
            pos_mean = model.q_pos_posterior(x0=pos0, xt=pos, t=t, batch=bl)
            pos_logvar = extract(model.posterior_logvar, t, bl)
            nonzero_mask = (1 - (t == 0).float())[bl].unsqueeze(-1)
            pos_next = pos_mean + nonzero_mask * (0.5 * pos_logvar).exp() * torch.randn_like(pos)

            # ── Apply guidance strategy ──
            guide_delta = torch.zeros_like(pos_next)

            if mode == "hard_fix" and anchor_targets is not None:
                for mol_i in range(n_curr):
                    mask_i = (bl == mol_i)
                    mol_indices = torch.where(mask_i)[0]
                    n_at_i = len(mol_indices)
                    for ai in anchor_indices:
                        if 0 <= ai < n_at_i and ai < len(anchor_targets):
                            abs_idx = mol_indices[ai]
                            guide_delta[abs_idx] = anchor_targets[ai].to(device) - pos_next[abs_idx]

            elif mode == "kinematic" and site_energy is not None and kin_scheduler is not None:
                t_norm = step_i / max(len(time_seq) - 1, 1)
                lam = kin_scheduler(t_norm)
                if isinstance(lam, torch.Tensor):
                    lam = lam.item()

                if lam > 0 and site_energy.n_sites > 0 and anchor_indices:
                    for mol_i in range(n_curr):
                        mask_i = (bl == mol_i)
                        mol_indices = torch.where(mask_i)[0]
                        n_at_i = len(mol_indices)
                        valid_anchors = [ai for ai in anchor_indices if 0 <= ai < n_at_i]
                        if not valid_anchors:
                            continue

                        anchor_pos = pos_next[mol_indices[valid_anchors]]
                        anchor_com = anchor_pos.mean(dim=0)

                        # Analytic site gradient at anchor CoM
                        site_centers = site_energy._site_centers.to(device)
                        sigma2 = 2.0 * 3.0 ** 2
                        rel_com = site_centers - anchor_com.unsqueeze(0)
                        dist_sq = (rel_com ** 2).sum(dim=-1)
                        gauss = torch.exp(-dist_sq / sigma2)

                        compat_mat = site_energy.compatibility_matrix.to(device)
                        env_idx = site_energy._site_env_indices.to(device)
                        best_compat = compat_mat[env_idx].max(dim=-1).values

                        weights = gauss * best_compat
                        if site_energy._site_confs is not None:
                            weights = weights * site_energy._site_confs.to(device)

                        grad = (weights.unsqueeze(-1) * rel_com / sigma2).sum(dim=0)
                        gnorm = grad.norm()
                        if gnorm > 1e-8:
                            grad = grad * (0.05 / gnorm)

                        correction = lam * grad
                        cnorm = correction.norm()
                        max_corr = 0.5
                        if cnorm > max_corr:
                            correction = correction * (max_corr / cnorm)

                        # Pure translation: same correction for all anchors
                        for ai in valid_anchors:
                            abs_idx = mol_indices[ai]
                            guide_delta[abs_idx] = correction

            pos_next = pos_next + guide_delta

            # ── KPE ──
            if track_kpe and pos_before is not None:
                delta_ode = pos_next - pos_before
                kpe_ode_total += (delta_ode ** 2).sum().item()
                kpe_guide_total += (guide_delta ** 2).sum().item()

            # Atom type update
            if v0 is not None:
                log_v_recon = F.log_softmax(v0, dim=-1)
                log_v_cur = index_to_log_onehot(v, model.num_classes)
                log_model_prob = model.q_v_posterior(log_v_recon, log_v_cur, t, bl)
                v = log_sample_categorical(log_model_prob)

            pos = pos_next

        # Restore offset
        pos_final = pos + offset[bl]

        # Unbatch
        pos_np = pos_final.detach().cpu().numpy().astype(np.float64)
        v_np = v.detach().cpu().numpy()
        ligand_cum = np.cumsum([0] + ligand_n_atoms)
        for k in range(n_curr):
            all_pos.append(pos_np[ligand_cum[k]:ligand_cum[k + 1]])
            all_v.append(v_np[ligand_cum[k]:ligand_cum[k + 1]])

    kpe_summary = {
        "kpe_ode_total": kpe_ode_total,
        "kpe_guide_total": kpe_guide_total,
        "kpe_ratio": kpe_guide_total / (kpe_ode_total + kpe_guide_total + 1e-8),
    }
    return all_pos, all_v, kpe_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruction and Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_mols(all_pos, all_v, output_dir, prefix="mol"):
    sdf_dir = Path(output_dir) / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    valid = []
    for i, (pos, v) in enumerate(zip(all_pos, all_v)):
        try:
            # OpenBabel requires float64 (double) coordinates
            pos64 = pos.astype(np.float64) if pos.dtype != np.float64 else pos
            atom_types = trans.get_atomic_number_from_index(v, mode='add_aromatic')
            aromatic = trans.is_aromatic_from_index(v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pos64, atom_types, aromatic)
            smiles = Chem.MolToSmiles(mol)
            if '.' not in smiles and len(Chem.GetMolFrags(mol)) == 1:
                mol.SetProp("_Name", f"{prefix}_{i:03d}")
                Chem.MolToMolFile(mol, str(sdf_dir / f"{prefix}_{i:03d}.sdf"))
                valid.append({"mol": mol, "idx": i, "pos": pos64, "v": v})
        except Exception:
            pass
    return valid


def compute_direct_occ(mols_data, site_map, threshold=2.5):
    """Fraction of molecules with >=1 atom within threshold of any HEW site."""
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return 0.0

    occupied = 0
    for md in mols_data:
        pos = md["pos"]   # [n_atoms, 3]
        for site in hew_sites:
            sc = np.array(site["center"])
            dists = np.linalg.norm(pos - sc, axis=-1)
            if dists.min() <= threshold:
                occupied += 1
                break

    return occupied / len(mols_data) if mols_data else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", required=True, choices=["3mfw", "6o4x", "2gni"])
    parser.add_argument("--mode", default="all",
                        choices=["unguided", "hard_fix", "kinematic", "all"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="experiments/targetdiff_replication")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--kinematic-lambda", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Only run 2 samples per mode")
    args = parser.parse_args()

    pocket = args.pocket
    cfg = POCKET_CONFIG[pocket]
    output_dir = Path(args.output_dir) / pocket
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load site map
    site_map_path = SITE_MAP_DIR / f"{pocket}_site_map.json"
    with open(site_map_path) as f:
        site_map = json.load(f)
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    print(f"Pocket {pocket}: {len(hew_sites)} HEW, {len(site_map['sites'])} total sites")

    # Build SiteCompatibilityEnergy
    site_energy = SiteCompatibilityEnergy(sigma_distance=3.0)
    if hew_sites:
        centers = torch.tensor([s["center"] for s in hew_sites], dtype=torch.float32)
        env_idx = torch.tensor([0] * len(hew_sites), dtype=torch.long)
        site_energy.register_sites(centers, env_idx)
    print(f"Site energy: {site_energy.n_sites} HEW registered")

    # Load model
    model, config, pf = load_model_and_config(args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{model.num_timesteps} timesteps, {model.num_classes} classes")

    # Load protein (use pocket PDB for speed — ~1000 atoms vs ~15000)
    pocket_pdb_path = f"/root/autodl-tmp/data/PDB/P-L/{cfg['year']}/{pocket}/{pocket}_pocket.pdb"
    protein_data = load_protein(pocket_pdb_path, pf)
    print(f"Pocket: {len(protein_data.protein_pos)} atoms (using pocket PDB)")

    # Define anchor atoms (first 4 atoms, placed near HEW sites via Phase 1 fragment)
    anchor_indices = [0, 1, 2, 3]
    if hew_sites:
        # Use the highest-confidence HEW site center as anchor target
        best_hew = sorted(hew_sites, key=lambda s: s.get("confidence", 0), reverse=True)[0]
        anchor_targets = torch.tensor([best_hew["center"]] * 4, dtype=torch.float32)
    else:
        anchor_targets = torch.zeros(4, 3)

    modes = ["unguided", "hard_fix", "kinematic"] if args.mode == "all" else [args.mode]
    n_each = 2 if args.dry_run else args.n_samples

    all_summaries = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"[{pocket}] Mode: {mode} ({n_each} molecules)")
        print("="*50)

        mode_dir = output_dir / mode
        t0 = time.time()

        # Determine anchor config per mode
        # For unguided, no anchor targets needed
        # For hard_fix, use the HEW site centers
        # For kinematic, use site_energy (no hard targets)
        at = None
        se = None
        kl = args.kinematic_lambda

        if mode == "hard_fix":
            at = anchor_targets
        elif mode == "kinematic":
            se = site_energy

        positions, types, kpe = generate_molecules_targetdiff(
            model, config, protein_data,
            n_samples=n_each,
            device=args.device,
            mode=mode,
            anchor_indices=anchor_indices,
            anchor_targets=at,
            site_energy=se,
            kinematic_lambda_max=kl,
            num_steps=args.num_steps,
            track_kpe=True,
            batch_size=args.batch_size,
        )

        elapsed = time.time() - t0

        # Reconstruct
        valid = reconstruct_mols(positions, types, mode_dir, prefix=mode)
        direct_occ = compute_direct_occ(
            [{"pos": p, "v": v} for p, v in zip(positions, types)], site_map
        )

        print(f"  Time: {elapsed:.1f}s ({elapsed/max(len(positions),1):.1f}s/mol)")
        print(f"  Valid: {len(valid)}/{len(positions)}")
        print(f"  DirectOcc: {direct_occ:.1%}")
        print(f"  KPE: ode={kpe['kpe_ode_total']:.1f}, guide={kpe['kpe_guide_total']:.1f}, "
              f"ratio={kpe['kpe_ratio']:.6f}")

        # Save
        torch.save({
            "positions": positions, "types": types, "kpe": kpe,
            "valid_count": len(valid), "direct_occ": direct_occ,
            "config": {"pocket": pocket, "mode": mode, "n_samples": n_each,
                       "num_steps": args.num_steps},
        }, mode_dir / "results.pt")

        all_summaries[mode] = {
            "direct_occ": direct_occ,
            "kpe_ratio": kpe["kpe_ratio"],
            "n_valid": len(valid),
            "n_total": len(positions),
            "time_sec": elapsed,
        }

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {pocket} (TargetDiff)")
    print("="*60)
    header = f"{'Condition':<15} {'DirectOcc':>10} {'KPE_ratio':>12} {'Valid':>8}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        s = all_summaries[mode]
        print(f"{mode:<15} {s['direct_occ']:>9.1%} {s['kpe_ratio']:>11.6f} "
              f"{s['n_valid']:>8}")

    summary = {
        "pocket": pocket,
        "generator": "TargetDiff",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "anchor_indices": anchor_indices,
        "conditions": all_summaries,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to {output_dir}/summary.json")

    # Print LaTeX table row
    print(f"\nLaTeX table fragment:")
    for mode in modes:
        s = all_summaries[mode]
        print(f"  {pocket} & {mode.capitalize()} & {s['direct_occ']:.1%} & "
              f"TBD & TBD & TBD & TBD \\\\")


if __name__ == "__main__":
    main()
