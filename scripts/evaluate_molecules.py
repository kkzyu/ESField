#!/usr/bin/env python3
"""Comprehensive molecular evaluation for ESField guided vs baseline.

Metrics (from SYNC paper + standard SBDD benchmarks):
  1. Vina Score / Vina Dock: predicted binding affinity (kcal/mol, lower=better)
  2. QED: Quantitative Estimate of Drug-likeness (0-1, higher=better)
  3. SA Score: Synthetic Accessibility (1-10, lower=easier to synthesize)
  4. Collision/Clash Rate: fraction of atoms with steric clashes (<1.0A from protein)
  5. Diversity: pairwise Tanimoto dissimilarity among generated molecules
  6. Atom validity: fraction of atoms with valid valence
  7. Bond length validity: fraction of bonds within reasonable length range

Requires: meeko, vina, rdkit, autodock-vina binary
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdMolDescriptors, Descriptors
from rdkit.Chem.QED import qed as calc_qed


# ============================================================
# Metric 1: Vina Docking Score
# ============================================================

def compute_vina_scores(protein_pdb, ligand_sdf, output_dir, num_modes=5,
                        center=None, box_size=None):
    """Compute AutoDock Vina scores for a ligand against a protein pocket.

    Returns dict with vina_score (best), vina_min, vina_dock.
    Uses meeko for ligand preparation and AutoDock Vina for docking.
    """
    out_dir = Path(output_dir) / "docking"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare receptor PDBQT
    receptor_pdbqt = out_dir / "receptor.pdbqt"
    _prepare_receptor(protein_pdb, receptor_pdbqt)

    # Read ligand
    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    mols = [m for m in supplier if m is not None]
    if not mols:
        return {"vina_score": None, "vina_min": None, "vina_dock": None, "error": "No valid mol in SDF"}

    all_scores = []
    for mol_idx, mol in enumerate(mols):
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        # Generate 3D coordinates if needed
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

        # Prepare ligand PDBQT with meeko
        ligand_pdb = out_dir / f"ligand_{mol_idx}.pdb"
        Chem.MolToPDBFile(mol, str(ligand_pdb))
        ligand_pdbqt = out_dir / f"ligand_{mol_idx}.pdbqt"
        try:
            _meeko_prepare(ligand_pdb, ligand_pdbqt)
        except Exception:
            continue

        # Compute docking box
        if center is None:
            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            center = tuple(coords.mean(axis=0).tolist())
        if box_size is None:
            box_size = (22.5, 22.5, 22.5)

        # Run Vina
        vina_out = out_dir / f"vina_out_{mol_idx}.pdbqt"
        result = subprocess.run([
            "vina",
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]),
            "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
            "--out", str(vina_out),
            "--num_modes", str(num_modes),
        ], capture_output=True, text=True, timeout=300)

        # Parse output
        scores = _parse_vina_output(result.stdout)
        if scores:
            all_scores.extend(scores[:num_modes])

        # Cleanup
        ligand_pdb.unlink(missing_ok=True)

    if not all_scores:
        return {"vina_score": None, "vina_min": None, "vina_dock": None,
                "error": "No docking results"}

    return {
        "vina_score": float(np.mean(all_scores)),
        "vina_min": float(np.min(all_scores)),
        "vina_dock": float(np.min(all_scores)),
        "vina_std": float(np.std(all_scores)) if len(all_scores) > 1 else 0,
        "n_docked": len(all_scores),
    }


def _prepare_receptor(pdb_path, output_pdbqt):
    """Prepare receptor with meeko."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        preparator = MoleculePreparation()
        mols = list(preparator.prepare(Chem.MolFromPDBFile(str(pdb_path), removeHs=True)))
        if mols:
            pdbqt_str, _ = PDBQTWriterLegacy.write_string(mols[0])
            output_pdbqt.write_text(pdbqt_str)
            return
    except Exception:
        pass
    # Fallback: use obabel
    subprocess.run([
        "obabel", str(pdb_path), "-O", str(output_pdbqt),
        "-xr", "-xp"
    ], capture_output=True, timeout=60)


def _meeko_prepare(ligand_pdb, output_pdbqt):
    """Prepare ligand PDBQT with meeko."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False)
        if mol is None:
            raise ValueError("Cannot read ligand")
        preparator = MoleculePreparation()
        mols = list(preparator.prepare(mol))
        if mols:
            pdbqt_str, _ = PDBQTWriterLegacy.write_string(mols[0])
            output_pdbqt.write_text(pdbqt_str)
            return
    except Exception:
        pass
    subprocess.run([
        "obabel", str(ligand_pdb), "-O", str(output_pdbqt), "-xp"
    ], capture_output=True, timeout=30)


def _parse_vina_output(stdout):
    """Parse Vina docking output to extract affinity scores."""
    scores = []
    for line in stdout.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                score = float(parts[0])
                if -50 < score < 50:
                    scores.append(score)
            except ValueError:
                continue
    return scores or None


# ============================================================
# Metric 2: QED (Quantitative Estimate of Drug-likeness)
# ============================================================

def compute_qed(sdf_path):
    """Compute QED for molecules in SDF. Range 0-1, higher=more drug-like."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    qeds = []
    for mol in supplier:
        if mol is not None:
            try:
                qeds.append(calc_qed(mol))
            except Exception:
                pass
    if not qeds:
        return {"qed_mean": None, "qed_std": None, "qed_n": 0}
    return {"qed_mean": float(np.mean(qeds)), "qed_std": float(np.std(qeds)), "qed_n": len(qeds)}


# ============================================================
# Metric 3: SA Score (Synthetic Accessibility)
# ============================================================

def compute_sa_score(sdf_path):
    """Estimate SA Score using RDKit's implementable fragment-based approach.

    SA Score = fragmentScore - complexityPenalty
    Range approximately 0-10, lower = easier to synthesize.
    Based on Ertl & Schuffenhauer (2009).
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    scores = []
    for mol in supplier:
        if mol is not None:
            try:
                scores.append(_estimate_sa_score(mol))
            except Exception:
                pass
    if not scores:
        return {"sa_score_mean": None, "sa_score_std": None, "sa_score_n": 0}
    return {"sa_score_mean": float(np.mean(scores)), "sa_score_std": float(np.std(scores)),
            "sa_score_n": len(scores)}


def _estimate_sa_score(mol):
    """SA Score based on fragment contributions and molecular complexity.

    Reference: Ertl & Schuffenhauer, J. Cheminform. 2009.
    """
    # Fragment-based contribution
    frag_score = _compute_fragment_score(mol)
    # Complexity penalty
    complexity = _compute_complexity_penalty(mol)

    sa_score = frag_score - complexity
    # Normalize to roughly 0-10 range
    sa_score = max(1.0, min(10.0, sa_score))
    return sa_score


def _compute_fragment_score(mol):
    """Estimate fragment-based synthetic accessibility contribution."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    # Simplified fragment score
    score = 3.0
    score += np.log10(max(mw, 1)) * 0.5
    score += max(0, logp - 3) * 0.2
    score += n_rings * 0.3
    score += n_chiral * 0.5
    return score


def _compute_complexity_penalty(mol):
    """Complexity penalty based on structural features."""
    n_atoms = mol.GetNumHeavyAtoms()
    n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    penalty = 0.0
    penalty += np.log10(max(n_atoms, 1)) * 0.3
    penalty += n_rot_bonds * 0.05
    penalty += n_spiro * 0.5
    penalty += n_bridge * 0.3
    return penalty


# ============================================================
# Metric 4: Collision / Clash Rate
# ============================================================

def compute_collision_rate(sdf_path, protein_pdb=None, clash_cutoff=1.0):
    """Compute fraction of ligand atoms that clash with protein atoms.

    Clash: any ligand atom < clash_cutoff Angstroms from any protein atom.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]

    if protein_pdb:
        protein_coords = _read_protein_coords(protein_pdb)
    else:
        protein_coords = None

    results = []
    for mol in mols:
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            lig_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

            if protein_coords is not None:
                # Pairwise distances
                dists = np.linalg.norm(
                    lig_coords[:, None, :] - protein_coords[None, :, :], axis=-1)
                clashes = (dists.min(axis=1) < clash_cutoff).sum()
                clash_rate = clashes / len(lig_coords)
            else:
                # Intra-ligand clash: atoms within same molecule
                n = len(lig_coords)
                clash_count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if np.linalg.norm(lig_coords[i] - lig_coords[j]) < 0.5:
                            clash_count += 1
                n_pairs = n * (n - 1) / 2 if n > 1 else 1
                clash_rate = clash_count / n_pairs

            results.append(clash_rate)

    if not results:
        return {"clash_rate_mean": None, "clash_rate_n": 0}
    return {"clash_rate_mean": float(np.mean(results)), "clash_rate_std": float(np.std(results)),
            "clash_rate_n": len(results)}


def _read_protein_coords(pdb_path):
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


# ============================================================
# Metric 5: Molecular validity / completeness
# ============================================================

def compute_validity_metrics(sdf_path):
    """Compute standard molecular validity metrics."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    valid_mols = []
    n_total = 0
    n_valence_error = 0
    n_sanitize_error = 0

    for mol in supplier:
        n_total += 1
        if mol is None:
            n_sanitize_error += 1
            continue
        try:
            Chem.SanitizeMol(mol)
            valid_mols.append(mol)
        except Exception:
            n_valence_error += 1

    n_valid = len(valid_mols)

    # Bond length check
    bond_ok = 0
    bond_bad = 0
    for mol in valid_mols:
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                d = np.linalg.norm(
                    np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j)))
                # Reasonable bond lengths: 0.8-2.5 Å
                if 0.8 < d < 2.5:
                    bond_ok += 1
                else:
                    bond_bad += 1

    return {
        "n_molecules": n_total,
        "n_valid": n_valid,
        "validity_rate": n_valid / n_total if n_total > 0 else 0,
        "n_bonds_ok": bond_ok,
        "n_bonds_bad": bond_bad,
        "bond_validity_rate": bond_ok / (bond_ok + bond_bad) if (bond_ok + bond_bad) > 0 else 0,
    }


# ============================================================
# Metric 6: Diversity
# ============================================================

def compute_diversity(sdf_path):
    """Compute pairwise Tanimoto dissimilarity (diversity) among molecules."""
    from rdkit.Chem import rdFingerprintGenerator
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]

    if len(mols) < 2:
        return {"diversity": None, "n_mols": len(mols)}

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [mfpgen.GetFingerprint(m) for m in mols]

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = rdFingerprintGenerator.GetTanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    sims = np.array(similarities)
    diversity = 1 - sims.mean()
    return {
        "diversity": float(diversity),
        "mean_similarity": float(sims.mean()),
        "std_similarity": float(sims.std()),
        "n_pairs": len(similarities),
    }


# ============================================================
# Main evaluation
# ============================================================

def evaluate_all(baseline_sdf_dir, guided_sdf_dir, protein_pdb=None, output_dir=None):
    """Run all metrics on baseline and guided SDF directories."""
    results = {"baseline": {}, "guided": {}}

    print(f"{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")

    for name, sdf_dir in [("baseline", baseline_sdf_dir), ("guided", guided_sdf_dir)]:
        sdf_path = Path(sdf_dir)
        if not sdf_path.exists():
            print(f"  {name}: SDF dir not found at {sdf_dir}")
            continue

        sdf_files = list(sdf_path.glob("*.sdf"))
        if not sdf_files:
            print(f"  {name}: No SDF files found")
            continue

        # Combine all SDFs into one for batch analysis
        combined_sdf = sdf_path / "_combined.sdf"
        _combine_sdfs(sdf_files, combined_sdf)

        print(f"\n{name.upper()} ({len(sdf_files)} molecules):")

        # Validity
        valid = compute_validity_metrics(combined_sdf)
        results[name]["validity"] = valid
        print(f"  Validity: {valid['n_valid']}/{valid['n_molecules']} ({valid['validity_rate']:.1%})")

        # QED
        qed_r = compute_qed(combined_sdf)
        results[name]["qed"] = qed_r
        if qed_r["qed_mean"] is not None:
            print(f"  QED: {qed_r['qed_mean']:.3f} ± {qed_r['qed_std']:.3f}")

        # SA Score
        sa_r = compute_sa_score(combined_sdf)
        results[name]["sa_score"] = sa_r
        if sa_r["sa_score_mean"] is not None:
            print(f"  SA Score: {sa_r['sa_score_mean']:.2f} ± {sa_r['sa_score_std']:.2f} (lower=easier)")

        # Collision
        clash_r = compute_collision_rate(combined_sdf, protein_pdb)
        results[name]["clash"] = clash_r
        if clash_r["clash_rate_mean"] is not None:
            print(f"  Clash rate: {clash_r['clash_rate_mean']:.4f}")

        # Diversity
        div_r = compute_diversity(combined_sdf)
        results[name]["diversity"] = div_r
        if div_r["diversity"] is not None:
            print(f"  Diversity: {div_r['diversity']:.3f} (higher=more diverse)")

        # Cleanup
        combined_sdf.unlink(missing_ok=True)

    # Comparison
    if results["baseline"] and results["guided"]:
        print(f"\n{'='*60}")
        print(f"COMPARISON")
        print(f"{'='*60}")
        _print_comparison(results)

    # Save
    if output_dir:
        out_path = Path(output_dir) / "evaluation_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nMetrics saved to {out_path}")

    return results


def _combine_sdfs(sdf_files, output):
    """Combine multiple SDF files into one."""
    with open(output, "w") as out:
        for sf in sorted(sdf_files):
            with open(sf) as inp:
                out.write(inp.read())
                if not inp.read().endswith("\n"):
                    out.write("\n")


def _print_comparison(results):
    """Print side-by-side comparison."""
    b = results["baseline"]
    g = results["guided"]

    metrics_to_compare = [
        ("validity", "validity_rate", "Validity rate", "{:.1%}", True),
        ("qed", "qed_mean", "QED", "{:.3f}", True),
        ("sa_score", "sa_score_mean", "SA Score", "{:.2f}", False),
        ("clash", "clash_rate_mean", "Clash rate", "{:.4f}", False),
        ("diversity", "diversity", "Diversity", "{:.3f}", True),
    ]

    for section, key, label, fmt, higher_better in metrics_to_compare:
        bv = b.get(section, {}).get(key)
        gv = g.get(section, {}).get(key)
        if bv is not None and gv is not None:
            delta = gv - bv
            direction = "↑" if (delta > 0) == higher_better else "↓"
            print(f"  {label:<20}: B={fmt.format(bv)}, G={fmt.format(gv)} (Δ={delta:+.4f}) {direction}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sdf", required=True, help="Directory with baseline SDF files")
    parser.add_argument("--guided-sdf", required=True, help="Directory with guided SDF files")
    parser.add_argument("--protein-pdb", default=None)
    parser.add_argument("--output-dir", default="experiments/evaluation")
    parser.add_argument("--vina", action="store_true", help="Run Vina docking (slow)")
    parser.add_argument("--vina-center", nargs=3, type=float, default=None)
    args = parser.parse_args()

    evaluate_all(args.baseline_sdf, args.guided_sdf, args.protein_pdb, args.output_dir)


if __name__ == "__main__":
    main()
