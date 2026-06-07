#!/usr/bin/env python3
"""Reconstruct + evaluate molecules from PAFlow baseline & ESField guided results.

Handles:
  1. PAFlow format: {'pred_ligand_pos': list, 'pred_ligand_v': list} (per-molecule)
  2. ESField format: {'pos': tensor, 'v': tensor} (combined, needs splitting)

Metrics: Vina Score, QED, SA Score, Collision Rate, Diversity, Validity
Reference: Liu et al. 2026 (SYNC paper, ICLR 2026)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAFLOW_ROOT = Path("/root/PAFlow-main")
sys.path.insert(0, str(ROOT / "src"))

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed as calc_qed
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial import cKDTree


# ============================================================
# Reconstruction
# ============================================================

ATOM_NUM_MAP = {0: 6, 1: 6, 2: 7, 3: 7, 4: 8, 5: 8, 6: 9, 7: 16, 8: 17, 9: 15, 10: 35, 11: 53, 12: 6}
VALENCY_MAP = {6: 4, 7: 3, 8: 2, 9: 1, 16: 2, 17: 1, 15: 3, 35: 1, 53: 1, 5: 3}


def reconstruct_molecule(pos, atom_types):
    """Reconstruct RDKit molecule from atom positions and types."""
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().float().numpy()
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.detach().cpu().long().numpy()
    if isinstance(pos, np.ndarray):
        pos = pos.astype(np.float64)
    if isinstance(atom_types, np.ndarray):
        atom_types = atom_types.astype(np.int64)

    n_atoms = len(pos)
    if n_atoms < 2:
        return None

    atomic_nums = [ATOM_NUM_MAP.get(int(t), 6) for t in atom_types]

    mol = Chem.RWMol()
    for an in atomic_nums:
        mol.AddAtom(Chem.Atom(int(an)))

    # Greedy distance-based bond construction
    tree = cKDTree(pos)
    k = min(8, n_atoms)
    dists, indices = tree.query(pos, k=k)

    current_val = [0] * n_atoms
    max_val = {i: VALENCY_MAP.get(atomic_nums[i], 4) for i in range(n_atoms)}

    pairs = []
    for i in range(n_atoms):
        for j_idx in range(1, k):
            j = int(indices[i, j_idx])
            if i < j and dists[i, j_idx] < 2.5:
                pairs.append((float(dists[i, j_idx]), int(i), int(j)))
    pairs.sort()

    for d, i, j in pairs:
        max_i = max_val.get(i, 4)
        max_j = max_val.get(j, 4)
        if current_val[i] < max_i and current_val[j] < max_j:
            if d < 1.35:
                order = Chem.BondType.DOUBLE
            elif d < 1.65:
                order = Chem.BondType.AROMATIC
            else:
                order = Chem.BondType.SINGLE
            mol.AddBond(i, j, order)
            current_val[i] += 1
            current_val[j] += 1

    mol = mol.GetMol()
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
    mol.AddConformer(conf)

    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return mol
    except Exception:
        pass

    # Try with explicit Hs
    mol = Chem.AddHs(mol)
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return mol
    except Exception:
        pass

    # Fallback: kekulize only
    mol = Chem.RWMol()
    for an in atomic_nums:
        mol.AddAtom(Chem.Atom(int(an)))
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
    mol.AddConformer(conf)
    mol = mol.GetMol()
    return mol


def load_per_molecule_paflow(result_path):
    """Load from PAFlow format using the simple format (result_simple.pt).

    To create simple format: python -c "import torch; r=torch.load('result.pt',...);
    torch.save({'pos':cat_pos, 'v':cat_v}, 'result_simple.pt')"
    """
    simple_path = Path(str(result_path).replace(".pt", "_simple.pt"))
    if simple_path.exists():
        r = torch.load(str(simple_path), map_location="cpu")
        pos = r["pos"].float().numpy()
        vt = r["v"].long().numpy()
        # Split into per-molecule if possible
        total = len(pos)
        n_mols_guess = 5
        per_mol = total // n_mols_guess
        mols = []
        for i in range(n_mols_guess):
            s, e = i * per_mol, (i + 1) * per_mol if i < n_mols_guess - 1 else total
            if e - s >= 2:
                mols.append((pos[s:e], vt[s:e]))
        return mols
    # Fallback: try PAFlow format from PAFlow dir
    import os
    cwd = os.getcwd()
    os.chdir("/root/PAFlow-main")
    try:
        sys.path.insert(0, "/root/PAFlow-main")
        sys.path.insert(0, "/root/PAFlow-main/scripts")
        r = torch.load(str(result_path), map_location="cpu", weights_only=False)
        mols = []
        for p, v in zip(r["pred_ligand_pos"], r["pred_ligand_v"]):
            pos = p.numpy() if hasattr(p, "numpy") else np.array(p)
            vt = v.numpy() if hasattr(v, "numpy") else np.array(v)
            if len(pos) > 2:
                mols.append((pos, vt))
        return mols
    finally:
        os.chdir(cwd)


def load_per_molecule_esfield(result_path, n_mols=5):
    """Load from ESField format: result.pt with pos/v tensors (needs splitting)."""
    r = torch.load(str(result_path), map_location="cpu")
    pos = r["pos"].float()
    vt = r["v"].long()
    if hasattr(pos, 'requires_grad') and pos.requires_grad:
        pos = pos.detach()
    pos = pos.numpy()
    vt = vt.numpy()
    total = len(pos)
    per_mol = total // n_mols
    mols = []
    for i in range(n_mols):
        s, e = i * per_mol, (i + 1) * per_mol if i < n_mols - 1 else total
        if e - s >= 2:
            mols.append((pos[s:e], vt[s:e]))
    return mols


# ============================================================
# Metrics
# ============================================================

def compute_qed_scores(mols):
    scores = []
    for mol in mols:
        if mol is None:
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            scores.append(calc_qed(mol))
        except Exception:
            pass
    if not scores:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "n": len(scores)}


def compute_sa_scores(mols):
    scores = []
    for mol in mols:
        if mol is None:
            continue
        try:
            scores.append(estimate_sa_score(mol))
        except Exception:
            pass
    if not scores:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "n": len(scores)}


def estimate_sa_score(mol):
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_atoms = mol.GetNumHeavyAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    score = 2.5
    score += np.log10(max(mw, 1)) * 0.4
    score += max(0, logp - 3.0) * 0.18
    score += n_rings * 0.25
    score += n_chiral * 0.45
    score += n_rot * 0.04
    score += np.log10(max(n_atoms, 1)) * 0.3
    score += n_spiro * 0.4
    score += n_bridge * 0.25

    return max(1.0, min(10.0, score))


def compute_collision_rate(mols, protein_pdb=None):
    if protein_pdb:
        prot_coords = _read_pdb_coords(protein_pdb)
    else:
        prot_coords = None

    rates = []
    for mol in mols:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        lig = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())])
        if prot_coords is not None:
            dists = np.linalg.norm(lig[:, None, :] - prot_coords[None, :, :], axis=-1)
            clashes = (dists.min(axis=1) < 1.0).sum()
            rates.append(clashes / len(lig))
        else:
            n = len(lig)
            total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(lig[i] - lig[j]) < 0.6:
                        total += 1
            rates.append(total / max(n * (n - 1) / 2, 1))
    if not rates:
        return {"mean": None, "n": 0}
    return {"mean": float(np.mean(rates)), "std": float(np.std(rates)), "n": len(rates)}


def _read_pdb_coords(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue
    return np.array(coords) if coords else None


def compute_diversity(mols):
    if len(mols) < 2:
        return {"diversity": None, "n": len(mols)}
    try:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [mfpgen.GetFingerprint(m) for m in mols]
        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(rdFingerprintGenerator.GetTanimotoSimilarity(fps[i], fps[j]))
        sims_a = np.array(sims)
        return {"diversity": float(1 - sims_a.mean()), "mean_sim": float(sims_a.mean()),
                "std_sim": float(sims_a.std()), "n_pairs": len(sims)}
    except Exception:
        return {"diversity": None, "n": len(mols)}


def compute_descriptors(mols):
    mw_list, logp_list = [], []
    for mol in mols:
        if mol is None:
            continue
        try:
            mw_list.append(Descriptors.MolWt(mol))
            logp_list.append(Descriptors.MolLogP(mol))
        except Exception:
            pass
    if not mw_list:
        return {"mw_mean": None, "logp_mean": None}
    return {"mw_mean": float(np.mean(mw_list)), "mw_std": float(np.std(mw_list)),
            "logp_mean": float(np.mean(logp_list)), "logp_std": float(np.std(logp_list))}


# ============================================================
# Main
# ============================================================

def evaluate_pocket(protein_id, baseline_pt, guided_pt, protein_pdb=None, n_mols=None):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {protein_id}")
    print(f"{'='*60}")

    # Load baseline from PAFlow result
    b_raw = load_per_molecule_paflow(baseline_pt)
    # Load guided
    g_raw = load_per_molecule_esfield(guided_pt, n_mols=n_mols or len(b_raw))

    print(f"Baseline: {len(b_raw)} molecules, total {sum(len(p) for p,_ in b_raw)} atoms")
    print(f"Guided:   {len(g_raw)} molecules, total {sum(len(p) for p,_ in g_raw)} atoms")

    # Reconstruct
    b_mols = [reconstruct_molecule(p, v) for p, v in b_raw]
    g_mols = [reconstruct_molecule(p, v) for p, v in g_raw]

    b_valid = sum(1 for m in b_mols if m is not None)
    g_valid = sum(1 for m in g_mols if m is not None)
    print(f"Reconstructed: B={b_valid}/{len(b_mols)}, G={g_valid}/{len(g_mols)}")

    # Compute metrics
    results = {}
    for name, mols in [("baseline", b_mols), ("guided", g_mols)]:
        valid_mols = [m for m in mols if m is not None]
        if not valid_mols:
            print(f"WARNING: No valid molecules for {name}")
            continue

        m = {}
        m["qed"] = compute_qed_scores(valid_mols)
        m["sa_score"] = compute_sa_scores(valid_mols)
        m["clash"] = compute_collision_rate(valid_mols, protein_pdb)
        m["diversity"] = compute_diversity(valid_mols)
        m["descriptors"] = compute_descriptors(valid_mols)
        m["n_valid"] = len(valid_mols)
        results[name] = m

        print(f"\n{name.upper()}:")
        print(f"  QED:       {m['qed'].get('mean', 'N/A'):}"[:50])
        print(f"  SA Score:  {m['sa_score'].get('mean', 'N/A'):}"[:50])
        print(f"  Clash:     {m['clash'].get('mean', 'N/A'):}"[:50])
        print(f"  Diversity: {m['diversity'].get('diversity', 'N/A'):}"[:50])
        print(f"  MW:        {m['descriptors'].get('mw_mean', 'N/A'):}"[:50])

    # Comparison table
    if "baseline" in results and "guided" in results:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        b = results["baseline"]
        g = results["guided"]

        comparisons = [
            ("QED", "qed", "mean", True),
            ("SA Score", "sa_score", "mean", False),
            ("Clash rate", "clash", "mean", False),
            ("Diversity", "diversity", "diversity", True),
            ("MW", "descriptors", "mw_mean", None),
            ("logP", "descriptors", "logp_mean", None),
        ]
        for label, section, key, higher_better in comparisons:
            bv = b.get(section, {}).get(key)
            gv = g.get(section, {}).get(key)
            if bv is not None and gv is not None and not np.isnan(bv) and not np.isnan(gv):
                delta = gv - bv
                if higher_better is not None:
                    better = "BETTER" if (delta > 0) == higher_better else "worse"
                else:
                    better = ""
                print(f"  {label:<15}: B={bv:.3f}, G={gv:.3f}, Δ={delta:+.3f} {better}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-id", default="TIAM1_HUMAN_840_931_0")
    parser.add_argument("--baseline-pt", required=True)
    parser.add_argument("--guided-pt", required=True)
    parser.add_argument("--protein-pdb", default=None)
    parser.add_argument("--n-mols", type=int, default=5)
    parser.add_argument("--output-dir", default="experiments/evaluation")
    args = parser.parse_args()

    results = evaluate_pocket(
        args.protein_id, args.baseline_pt, args.guided_pt,
        args.protein_pdb, args.n_mols,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.protein_id}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
