#!/usr/bin/env python3
"""Pharmacophore-constrained generation experiment (Exp 3.5).

For each pocket, generates molecules under 4 conditions using pharmacophore
feature points as guidance targets (no Phase 1, single-stage only).

Conditions:
  - unguided:  pure DrugFlow (no guidance)
  - hard_fix:  anchor atoms hard-fixed to ref-ligand atoms nearest pharm points
  - full_grad: per-atom gradient toward all pharm points (no CoM)
  - kag:       CoM projection guidance toward pharm points

Usage:
    python scripts/run_pharmacophore_experiment.py --pockets 3mfw --n-mols 10
    python scripts/run_pharmacophore_experiment.py --all --n-mols 50
"""

from __future__ import annotations

import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"
DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
sys.path.insert(0, os.path.join(DRUGFLOW_DIR, "src"))
sys.path.insert(0, DRUGFLOW_DIR)

from src.model import lightning as lmod
from src.data.data_utils import process_raw_pair, TensorDict
from src.data.dataset import ProcessedLigandPocketDataset
from torch.utils.data import DataLoader
from functools import partial
from Bio.PDB import PDBParser

from guidance.pharmacophore_guidance import (
    build_pharmacophore_site_energy,
    PharmacophoreGuideFn,
    load_pharm_site_map,
)
from guidance.hard_fix import (
    HardFixCallback, patch_drugflow_hardfix, patch_drugflow_sample_post_step,
)
from guidance.kinematic_anchor import KinematicAnchorGuidance

POCKETS_CONFIG = {
    "3mfw": "2001-2010", "2gni": "2001-2010", "2gqn": "2001-2010",
    "2jke": "2001-2010", "6o4x": "2011-2019", "6phx": "2011-2019",
}
DATA_BASE = Path("/root/autodl-tmp/data/PDB/P-L")
PHARM_DIR = ROOT / "results/exp3.5_pharmacophore/pharm_sites"
OUTPUT_BASE = ROOT / "results/exp3.5_pharmacophore/generated"


def load_model(device="cuda:0"):
    warnings.filterwarnings("ignore")
    _o = torch.load
    torch.load = lambda *a, **kw: _o(*a, **{**kw, "weights_only": False})
    try:
        m = lmod.DrugFlow.load_from_checkpoint(DRUGFLOW_CKPT, map_location=device)
    finally:
        torch.load = _o
    return m.to(device).eval()


def process_protein(protein_pdb, ref_ligand_sdf, model):
    pdb_model = PDBParser(QUIET=True).get_structure("", protein_pdb)[0]
    rdmol = Chem.SDMolSupplier(ref_ligand_sdf)[0]
    ref_size = rdmol.GetNumAtoms()
    ligand_raw, pocket_raw = process_raw_pair(
        pdb_model, rdmol, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True)
    ligand_raw["name"] = "ligand"
    collate = partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None)
    data = next(iter(DataLoader(
        [{"ligand": ligand_raw, "pocket": pocket_raw}], batch_size=1, collate_fn=collate)))
    return data, ref_size


# ═══════════════════════════════════════════════════════════════════════════
# Generate molecules with pharmacophore guidance
# ═══════════════════════════════════════════════════════════════════════════

def gen_unguided(model, protein_data, n, full_size, device):
    mols = []
    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(protein_data, n_samples=n, timesteps=100, num_nodes=full_size)
    for m in rdmols:
        if m is not None:
            try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except: pass
            mols.append(m)
    return mols, time.time() - t0


def gen_full_grad(model, protein_data, n, full_size, device, pharm_map):
    energy = build_pharmacophore_site_energy(pharm_map, sigma_distance=3.0).to(device)
    guide = PharmacophoreGuideFn(energy, lambda_guide=1.0).to(device)
    mols = []
    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(protein_data, n_samples=n, timesteps=100, num_nodes=full_size, guide_log_prob=guide)
    for m in rdmols:
        if m is not None:
            try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except: pass
            mols.append(m)
    return mols, time.time() - t0


def gen_hard_fix(model, protein_data, n, full_size, device, pharm_map, anchor_coords):
    patch_drugflow_hardfix()
    energy = build_pharmacophore_site_energy(pharm_map, sigma_distance=3.0).to(device)
    n_anchors = len(anchor_coords)
    anchor_tensor = torch.tensor(anchor_coords, dtype=torch.float32)

    # Use pharmacophore guide
    guide = PharmacophoreGuideFn(energy, lambda_guide=0.1).to(device)

    cb = HardFixCallback(
        anchor_indices=list(range(n_anchors)),
        anchor_coords=anchor_tensor,
        fix_coords=True, fix_types=False,
    )
    patch_drugflow_sample_post_step(model)

    mols = []
    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(
            protein_data, n_samples=n, timesteps=100, num_nodes=full_size,
            guide_log_prob=guide, post_step_callback=cb,
        )
    for m in rdmols:
        if m is not None:
            try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except: pass
            mols.append(m)
    return mols, time.time() - t0


def gen_kag_pharm(model, protein_data, n, full_size, device, pharm_map):
    """KAG-style CoM projection toward pharmacophore sites (single-stage, no anchors)."""
    patch_drugflow_hardfix()
    energy = build_pharmacophore_site_energy(pharm_map, sigma_distance=3.0).to(device)

    # Use CoM projection on ALL atoms (single-stage KAG)
    cb = KinematicAnchorGuidance(
        anchor_indices=list(range(full_size)),  # all atoms
        site_energy=energy,
        total_steps=100,
        lambda_max=1.0,
        profile="quadratic",
        track_kpe=True,
    )
    patch_drugflow_sample_post_step(model)

    mols = []
    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(
            protein_data, n_samples=n, timesteps=100, num_nodes=full_size,
            post_step_callback=cb,
        )
    for m in rdmols:
        if m is not None:
            try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except: pass
            mols.append(m)
    return mols, time.time() - t0


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation (pharmacophore-specific metrics)
# ═══════════════════════════════════════════════════════════════════════════

def compute_pharm_metrics(mols, pharm_map):
    """Compute pharmacophore proximity metrics."""
    sites = pharm_map.get("sites", [])
    if not sites:
        return {}

    centers = np.array([s["center"] for s in sites])
    ptypes = [s.get("pharm_type", "hydrophobic") for s in sites]

    results = []
    for mol in mols:
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        conf = mol.GetConformer()
        if conf is None:
            continue

        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
        centroid = coords.mean(axis=0)

        # Min/avg distance from centroid to pharm points
        cdists = np.linalg.norm(centers - centroid[None, :], axis=1)
        min_dist = float(cdists.min())
        avg_dist = float(cdists.mean())

        # Pharm-COS: for each pharm point, find best-matching atom
        from metrics_new import compute_cos, env_to_idx, classify_hew_environment
        # Simplified: compute COS-like score with σ=1.5
        cos_vals = []
        for j in range(len(centers)):
            rel = coords - centers[j][None, :]
            dists = np.linalg.norm(rel, axis=1)
            gauss = np.exp(-dists**2 / (2 * 1.5**2))
            # Compatibility: simplified — just use gauss weight × type bonus
            # For HBD: N_donor gets +1, for HBA: O_acceptor gets +1
            type_bonus = np.zeros(mol.GetNumAtoms())
            for i, atom in enumerate(mol.GetAtoms()):
                anum = atom.GetAtomicNum()
                pt = ptypes[j]
                if pt == "hbd" and anum == 7:
                    type_bonus[i] = 1.0
                elif pt == "hba" and anum == 8:
                    type_bonus[i] = 1.0
                elif pt == "hydrophobic" and anum == 6 and not atom.GetIsAromatic():
                    type_bonus[i] = 1.0
                elif pt == "aromatic" and atom.GetIsAromatic():
                    type_bonus[i] = 1.0
                elif pt == "pos_ion" and anum == 7:
                    type_bonus[i] = 0.5
                elif pt == "neg_ion" and anum == 8:
                    type_bonus[i] = 0.5
            score = float((gauss * type_bonus).max())
            cos_vals.append(score)

        cos_vals = np.array(cos_vals)

        try:
            qed_val = QED.qed(mol)
        except:
            qed_val = float("nan")

        try:
            from rdkit.Contrib.SA_Score import sascorer
            sa_val = sascorer.calculateScore(mol)
        except:
            sa_val = float("nan")

        results.append({
            "min_dist_pharm": min_dist,
            "avg_dist_pharm": avg_dist,
            "avg_pharm_COS": float(cos_vals.mean()),
            "max_pharm_COS": float(cos_vals.max()),
            "QED": qed_val,
            "SA": sa_val,
            "n_atoms": mol.GetNumAtoms(),
        })

    if not results:
        return {"n": 0}

    stats = {}
    for k in ["min_dist_pharm", "avg_dist_pharm", "avg_pharm_COS", "max_pharm_COS", "QED", "SA"]:
        vals = [r[k] for r in results if not np.isnan(r.get(k, float("nan")))]
        if vals:
            stats[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    return {"n": len(results), "statistics": stats, "per_mol": results}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets", default="3mfw", help="Comma-separated pocket IDs")
    parser.add_argument("--conditions", default="unguided,hard_fix,full_grad,kag")
    parser.add_argument("--n-mols", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pockets = [p.strip() for p in args.pockets.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    print("Loading DrugFlow...")
    model = load_model(args.device)

    for pocket in pockets:
        year = POCKETS_CONFIG[pocket]
        ppdb = DATA_BASE / year / pocket / f"{pocket}_protein.pdb"
        rlig = DATA_BASE / year / pocket / f"{pocket}_ligand.sdf"
        pmap_path = PHARM_DIR / f"{pocket}_pharm.json"

        if not pmap_path.exists():
            print(f"  ⚠ No pharm map for {pocket}")
            continue

        pharm_map = load_pharm_site_map(pmap_path)
        n_pharm = len(pharm_map.get("sites", []))
        print(f"\n{'='*50}\n  {pocket} — {n_pharm} pharm features\n{'='*50}")

        data, ref_size = process_protein(str(ppdb), str(rlig), model)
        full_size = ref_size
        protein_data = {
            "ligand": TensorDict(**data["ligand"]).to(args.device),
            "pocket": TensorDict(**data["pocket"]).to(args.device),
        }

        # Get anchor positions from reference ligand (closest atom to each pharm point)
        ref_mol = Chem.SDMolSupplier(str(rlig))[0]
        ref_conf = ref_mol.GetConformer()
        pharm_centers = np.array([s["center"] for s in pharm_map["sites"]])
        anchor_coords = []
        for pc in pharm_centers:
            best_d = float("inf")
            best_pos = None
            for i in range(ref_mol.GetNumAtoms()):
                pos = np.array(ref_conf.GetAtomPosition(i))
                d = np.linalg.norm(pos - pc)
                if d < best_d:
                    best_d = d
                    best_pos = pos
            if best_pos is not None:
                anchor_coords.append(best_pos.tolist())
        anchor_coords = anchor_coords[:min(len(anchor_coords), full_size)]

        for cond in conditions:
            out_dir = OUTPUT_BASE / pocket / cond
            out_dir.mkdir(parents=True, exist_ok=True)
            sdf_path = out_dir / "molecules.sdf"
            if sdf_path.exists():
                print(f"  {cond}: exists, skip")
                continue

            print(f"  {cond} ({args.n_mols} mols)...", end=" ", flush=True)
            try:
                if cond == "unguided":
                    mols, t = gen_unguided(model, protein_data, args.n_mols, full_size, args.device)
                elif cond == "full_grad":
                    mols, t = gen_full_grad(model, protein_data, args.n_mols, full_size, args.device, pharm_map)
                elif cond == "hard_fix":
                    mols, t = gen_hard_fix(model, protein_data, args.n_mols, full_size, args.device, pharm_map, anchor_coords[:4])
                elif cond == "kag":
                    mols, t = gen_kag_pharm(model, protein_data, args.n_mols, full_size, args.device, pharm_map)
                else:
                    continue
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # Save SDF
            w = Chem.SDWriter(str(sdf_path))
            w.SetKekulize(False)
            for m in mols:
                try: w.write(m)
                except: pass
            w.close()

            # Evaluate
            metrics = compute_pharm_metrics(mols, pharm_map)
            meta = {"pocket": pocket, "condition": cond, "n": len(mols), "time_s": t, "statistics": metrics.get("statistics", {})}
            json.dump(meta, open(out_dir / "metadata.json", "w"), indent=2, default=str)

            s = metrics.get("statistics", {})
            print(f"{len(mols)} mols, {t:.0f}s")
            for k in ["min_dist_pharm", "avg_pharm_COS", "QED", "SA"]:
                if k in s:
                    print(f"    {k}: {s[k]['mean']:.3f}±{s[k]['std']:.3f}")

    print("\n✓ Pharmacophore experiment complete")


if __name__ == "__main__":
    main()
