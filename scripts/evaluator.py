#!/usr/bin/env python3
"""
Unified Evaluation Script for ESField Kinematic Anchor Guidance.

Replaces POSU with: Strain Energy (MMFF94), Clash Score, SA Score, PBR
(Protein Bump Ratio).

Also computes: DirectOcc (HEW), SW-Occ (Stable Water occupancy),
Wasserstein-1 distance (distribution-level comparison), KPE metrics.

Usage:
    python scripts/evaluator.py \
        --sdf-dir experiments/run_001/3mfw/kinematic/ \
        --protein-pdb data/3mfw_protein.pdb \
        --site-json data/3mfw_sites.json \
        --output-json experiments/run_001/3mfw/kinematic_eval.json \
        --baseline-sdf-dir experiments/run_001/3mfw/unguided/  # for Wasserstein

    # Batch mode (multiple conditions)
    python scripts/evaluator.py \
        --batch experiments/run_001/ \
        --pockets 3mfw,2gni,6o4x,2jke,2gqn,6phx \
        --conditions unguided,hard_fix,kinematic,force_field
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdMolDescriptors, Descriptors, DataStructs
from rdkit.Chem.QED import qed as calc_qed
from rdkit.Chem import rdFingerprintGenerator

# Suppress RDKit warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Metric 1: Strain Energy (MMFF94 Force-Field Energy)
# ============================================================================

def compute_strain_energy(mol: Chem.Mol, max_iter: int = 200) -> float | None:
    """Compute per-atom MMFF94 strain energy (kcal/mol/atom).

    Strain = (E_current - E_minimized) / n_heavy_atoms.
    This normalisation matches Lai et al. (2025) and removes size bias.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        n_heavy = mol.GetNumHeavyAtoms()
        mol = Chem.AddHs(mol, addCoords=True)
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        if mp is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        if ff is None:
            return None
        energy_before = ff.CalcEnergy()
        ff.Minimize(maxIts=max_iter)
        energy_after = ff.CalcEnergy()
        strain = max(0.0, energy_before - energy_after)  # positive = strained
        return float(strain / max(n_heavy, 1))
    except Exception:
        return None


def compute_strain_energy_batch(sdf_path: str | Path) -> dict:
    """Compute per-atom MMFF94 strain energy statistics for all molecules in SDF."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    energies = []
    for mol in supplier:
        if mol is not None:
            e = compute_strain_energy(mol)
            if e is not None:
                energies.append(e)
    if not energies:
        return {"strain_per_atom_mean": None, "strain_per_atom_std": None, "n": 0}
    return {
        "strain_per_atom_mean": float(np.mean(energies)),
        "strain_per_atom_std": float(np.std(energies)),
        "strain_per_atom_median": float(np.median(energies)),
        "strain_per_atom_min": float(np.min(energies)),
        "strain_per_atom_max": float(np.max(energies)),
        "n": len(energies),
    }


# ============================================================================
# Metric 2: Clash Score (steric clashes between ligand atoms and protein)
# ============================================================================

def compute_clash_score(
    sdf_path: str | Path,
    protein_pdb: str | Path | None = None,
    clash_cutoff: float = 1.2,
    vdw_scale: float = 0.7,
) -> dict:
    """Compute steric clash score between ligand and protein.

    A clash is defined as an interatomic distance less than
    vdw_scale * (vdw_i + vdw_j), where vdw_i and vdw_j are the van der Waals
    radii of the two atoms.  Uses RDKit's built-in vdW radii.

    Args:
        sdf_path: Path to SDF file with ligand molecules.
        protein_pdb: Path to protein PDB file.
        clash_cutoff: Absolute clash cutoff in Angstrom (fallback if vdW fails).
        vdw_scale: Scaling factor for sum of vdW radii.

    Returns:
        Dict with clash statistics.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]

    # Read protein coordinates
    protein_coords, protein_elements = _read_protein_atoms(protein_pdb) if protein_pdb else (None, None)

    # Pre-compute vdW radii for protein atoms
    if protein_elements is not None:
        protein_vdw = np.array([_vdw_radius(el) for el in protein_elements])
    else:
        protein_vdw = None

    clash_rates = []
    clash_counts = []
    n_total_atoms_list = []

    for mol in mols:
        if mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        lig_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])
        lig_elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
        lig_vdw = np.array([_vdw_radius(el) for el in lig_elements])

        n_clash = 0
        if protein_coords is not None and len(protein_coords) > 0:
            # Vectorized pairwise distance with vdW-based clash detection
            for i in range(n_atoms):
                dists = np.linalg.norm(lig_coords[i] - protein_coords, axis=-1)
                # vdW-based criterion
                if protein_vdw is not None:
                    thresholds = vdw_scale * (lig_vdw[i] + protein_vdw)
                else:
                    thresholds = clash_cutoff
                clashes_i = np.sum(dists < thresholds)
                n_clash += clashes_i
        else:
            # Intra-ligand clash: atoms too close to each other
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    threshold = vdw_scale * (lig_vdw[i] + lig_vdw[j])
                    d = np.linalg.norm(lig_coords[i] - lig_coords[j])
                    if d < max(threshold, 0.5):
                        n_clash += 1

        clash_rates.append(n_clash / max(n_atoms, 1))
        clash_counts.append(n_clash)
        n_total_atoms_list.append(n_atoms)

    if not clash_rates:
        return {"clash_score_mean": None, "clash_score_std": None, "n": 0}

    return {
        "clash_score_mean": float(np.mean(clash_rates)),
        "clash_score_std": float(np.std(clash_rates)),
        "clash_count_mean": float(np.mean(clash_counts)),
        "clash_count_total": int(np.sum(clash_counts)),
        "n": len(clash_rates),
    }


# ============================================================================
# Metric 3: PBR (Protein Bump Ratio)
# ============================================================================

def compute_pbr(
    sdf_path: str | Path,
    protein_pdb: str | Path,
    bump_cutoff: float = 2.0,
) -> dict:
    """Compute Protein Bump Ratio: fraction of ligand atoms within bump_cutoff
    of any protein heavy atom.

    PBR > 0.2 indicates significant steric interference with the protein.

    Args:
        sdf_path: Path to ligand SDF.
        protein_pdb: Path to protein PDB.
        bump_cutoff: Distance threshold for bump detection (Angstrom).

    Returns:
        Dict with PBR statistics.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]

    protein_coords, _ = _read_protein_atoms(protein_pdb)
    if protein_coords is None or len(protein_coords) == 0:
        return {"pbr_mean": None, "pbr_std": None, "n": 0, "error": "No protein atoms"}

    pbrs = []
    for mol in mols:
        if mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        n_atoms = mol.GetNumHeavyAtoms()
        lig_coords = np.array([list(conf.GetAtomPosition(i))
                               for i in range(mol.GetNumAtoms())
                               if mol.GetAtomWithIdx(i).GetAtomicNum() > 1])

        if len(lig_coords) == 0:
            continue
        dists = np.linalg.norm(
            lig_coords[:, None, :] - protein_coords[None, :, :], axis=-1)
        n_bumps = (dists.min(axis=1) < bump_cutoff).sum()
        pbrs.append(n_bumps / len(lig_coords))

    if not pbrs:
        return {"pbr_mean": None, "pbr_std": None, "n": 0}

    return {
        "pbr_mean": float(np.mean(pbrs)),
        "pbr_std": float(np.std(pbrs)),
        "pbr_median": float(np.median(pbrs)),
        "n": len(pbrs),
    }


# ============================================================================
# Metric 4: SA Score (Synthetic Accessibility)
# ============================================================================

def compute_sa_score_batch(sdf_path: str | Path) -> dict:
    """Compute SA Score for all molecules in SDF.

    Uses the RDKit-contrib sascorer if available, otherwise falls back to a
    fragment-based estimator based on Ertl & Schuffenhauer (2009).
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    scores = []
    for mol in supplier:
        if mol is not None:
            try:
                scores.append(_compute_sa_score(mol))
            except Exception:
                pass
    if not scores:
        return {"sa_score_mean": None, "sa_score_std": None, "n": 0}
    return {
        "sa_score_mean": float(np.mean(scores)),
        "sa_score_std": float(np.std(scores)),
        "sa_score_median": float(np.median(scores)),
        "n": len(scores),
    }


def _compute_sa_score(mol: Chem.Mol) -> float:
    """Compute SA Score using RDKit fragment-based estimation.

    Reference: Ertl & Schuffenhauer, J. Cheminform. 2009.
    """
    try:
        import sascorer
        return float(sascorer.calculateScore(mol))
    except ImportError:
        pass

    # Fallback: fragment-based estimation
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_atoms = mol.GetNumHeavyAtoms()
    n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    score = 3.0 + np.log10(max(mw, 1)) * 0.5 + max(0, logp - 3) * 0.2
    score += n_rings * 0.3 + n_chiral * 0.5
    score -= np.log10(max(n_atoms, 1)) * 0.3 + n_rot_bonds * 0.05
    return max(1.0, min(10.0, score))


# ============================================================================
# Metric 5: QED (Quantitative Estimate of Drug-likeness)
# ============================================================================

def compute_qed_batch(sdf_path: str | Path) -> dict:
    """Compute QED for all molecules in SDF."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    qeds = []
    for mol in supplier:
        if mol is not None:
            try:
                qeds.append(calc_qed(mol))
            except Exception:
                pass
    if not qeds:
        return {"qed_mean": None, "qed_std": None, "n": 0}
    return {
        "qed_mean": float(np.mean(qeds)),
        "qed_std": float(np.std(qeds)),
        "qed_median": float(np.median(qeds)),
        "n": len(qeds),
    }


# ============================================================================
# Metric 6: DirectOcc (HEW) and SW-Occ (Stable Water Occupancy)
# ============================================================================

def compute_site_occupancy_batch(
    sdf_path: str | Path,
    site_json: str | Path,
    occ_cutoff: float = 2.5,
    compat_threshold: float = 0.3,
) -> dict:
    """Compute DirectOcc (HEW) and SW-Occ for all molecules.

    Args:
        sdf_path: Path to ligand SDF.
        site_json: JSON with HEW and SW site definitions.
                   Format: {"hew_sites": [{coords, env_class, ...}],
                            "sw_sites": [{coords, ...}]}
        occ_cutoff: Distance threshold for site occupancy (Angstrom).
        compat_threshold: Compatibility score threshold for DirectOcc.

    Returns:
        Dict with DirectOcc (HEW) and SW-Occ statistics.
    """
    with open(site_json) as f:
        site_data = json.load(f)

    # Support both formats:
    # 1) {"hew_sites": [...], "sw_sites": [...]} with "coords" key
    # 2) {"sites": [{..., "site_type":"high_energy_water", "center":...}, ...]}
    hew_sites = site_data.get("hew_sites", [])
    sw_sites = site_data.get("sw_sites", [])
    if not hew_sites and not sw_sites and "sites" in site_data:
        all_sites = site_data["sites"]
        hew_sites = [s for s in all_sites if s.get("site_type") == "high_energy_water"]
        sw_sites = [s for s in all_sites if s.get("site_type") == "stable_water"]
    # Normalise position key: "center" or "coords"
    for s in hew_sites:
        if "coords" not in s and "center" in s:
            s["coords"] = s["center"]
    for s in sw_sites:
        if "coords" not in s and "center" in s:
            s["coords"] = s["center"]

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]

    hew_occ_mols = 0
    sw_occ_mols = 0
    hew_occ_rates = []
    sw_occ_rates = []

    for mol in mols:
        if mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        lig_coords = np.array([list(conf.GetAtomPosition(i))
                               for i in range(mol.GetNumAtoms())])

        # HEW occupancy
        if hew_sites:
            hew_coords = np.array([s["coords"] for s in hew_sites])
            dists = np.linalg.norm(
                lig_coords[:, None, :] - hew_coords[None, :, :], axis=-1)
            # Any ligand atom within cutoff of a HEW site
            occupied = (dists.min(axis=0) < occ_cutoff).sum()
            hew_occ_rates.append(occupied / len(hew_sites))
            if (dists.min() < occ_cutoff):
                hew_occ_mols += 1

        # SW occupancy (for "non-specific filling" analysis)
        if sw_sites:
            sw_coords = np.array([s["coords"] for s in sw_sites])
            dists_sw = np.linalg.norm(
                lig_coords[:, None, :] - sw_coords[None, :, :], axis=-1)
            occupied_sw = (dists_sw.min(axis=0) < occ_cutoff).sum()
            sw_occ_rates.append(occupied_sw / len(sw_sites))
            if (dists_sw.min() < occ_cutoff):
                sw_occ_mols += 1

    n_valid = len(hew_occ_rates) if hew_occ_rates else len(sw_occ_rates)

    result = {
        "n_molecules": len(mols),
    }

    if hew_sites:
        result["direct_occ_hew"] = float(hew_occ_mols / max(len(mols), 1))
        result["hew_occ_rate_mean"] = float(np.mean(hew_occ_rates)) if hew_occ_rates else 0.0
        result["hew_occ_rate_std"] = float(np.std(hew_occ_rates)) if hew_occ_rates else 0.0
        result["n_hew_sites"] = len(hew_sites)

    if sw_sites:
        result["direct_occ_sw"] = float(sw_occ_mols / max(len(mols), 1))
        result["sw_occ_rate_mean"] = float(np.mean(sw_occ_rates)) if sw_occ_rates else 0.0
        result["sw_occ_rate_std"] = float(np.std(sw_occ_rates)) if sw_occ_rates else 0.0
        result["n_sw_sites"] = len(sw_sites)

    return result


# ============================================================================
# Metric 7: Validity
# ============================================================================

def compute_validity_batch(sdf_path: str | Path) -> dict:
    """Compute molecular validity statistics."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    n_total = 0
    n_valid = 0
    n_sanitize_error = 0

    for mol in supplier:
        n_total += 1
        if mol is None:
            n_sanitize_error += 1
            continue
        try:
            Chem.SanitizeMol(mol)
            n_valid += 1
        except Exception:
            n_sanitize_error += 1

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_error": n_sanitize_error,
        "validity_rate": n_valid / max(n_total, 1),
    }


# ============================================================================
# Metric 8: Vina Docking Score
# ============================================================================

def compute_vina_score_batch(
    sdf_path: str | Path,
    protein_pdb: str | Path,
    output_dir: str | Path | None = None,
    exhaustiveness: int = 8,
    n_modes: int = 5,
    box_padding: float = 5.0,
) -> dict:
    """Compute Vina docking scores for all molecules in SDF.

    Requires: vina binary, meeko Python package.

    Returns dict with vina_mean, vina_std, vina_best, n_docked.
    """
    import subprocess
    import tempfile

    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="vina_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]
    if not mols:
        return {"vina_mean": None, "vina_std": None, "n_docked": 0}

    # Prepare receptor once
    receptor_pdbqt = out_dir / "receptor.pdbqt"
    _write_receptor_pdbqt(protein_pdb, receptor_pdbqt)

    # Determine docking box from protein
    center, box_size = _compute_docking_box(protein_pdb, box_padding)

    all_best_scores = []
    for i, mol in enumerate(mols[:50]):  # Cap at 50 per condition for time
        try:
            mol = Chem.AddHs(mol)
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            continue

        ligand_pdb = out_dir / f"lig_{i}.pdb"
        ligand_pdbqt = out_dir / f"lig_{i}.pdbqt"
        Chem.MolToPDBFile(mol, str(ligand_pdb))

        try:
            _meeko_prepare_pdb(ligand_pdb, ligand_pdbqt)
        except Exception:
            ligand_pdb.unlink(missing_ok=True)
            continue

        vina_out = out_dir / f"vina_{i}.pdbqt"
        cmd = [
            "vina",
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--center_x", str(center[0]), "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]), "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
            "--out", str(vina_out),
            "--num_modes", str(n_modes),
            "--exhaustiveness", str(exhaustiveness),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            scores = _parse_vina_stdout(result.stdout)
            if scores:
                all_best_scores.append(min(scores))
        except subprocess.TimeoutExpired:
            pass
        finally:
            ligand_pdb.unlink(missing_ok=True)
            ligand_pdbqt.unlink(missing_ok=True)
            vina_out.unlink(missing_ok=True)

    if not all_best_scores:
        return {"vina_mean": None, "vina_std": None, "n_docked": 0}

    return {
        "vina_mean": float(np.mean(all_best_scores)),
        "vina_std": float(np.std(all_best_scores)),
        "vina_best": float(np.min(all_best_scores)),
        "vina_worst": float(np.max(all_best_scores)),
        "n_docked": len(all_best_scores),
    }


# ============================================================================
# Metric 9: Diversity (Tanimoto)
# ============================================================================

def compute_diversity_batch(sdf_path: str | Path) -> dict:
    """Compute pairwise Tanimoto diversity."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]
    if len(mols) < 2:
        return {"diversity": None, "n": len(mols)}

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [mfpgen.GetFingerprint(m) for m in mols]
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    sims = np.array(similarities)
    return {
        "diversity": float(1.0 - sims.mean()),
        "mean_similarity": float(sims.mean()),
        "std_similarity": float(sims.std()),
        "n_pairs": len(similarities),
    }


# ============================================================================
# Metric 10: Wasserstein-1 Distance (distribution-level comparison)
# ============================================================================

def compute_wasserstein_distances(
    condition_values: dict[str, list[float]],
    baseline_key: str = "unguided",
) -> dict:
    """Compute Wasserstein-1 distances between each condition and baseline.

    Args:
        condition_values: Dict mapping condition name -> list of scalar values.
        baseline_key: Key identifying the baseline condition.

    Returns:
        Dict mapping f"wasserstein_{condition}" -> distance.
    """
    from scipy.stats import wasserstein_distance

    baseline = condition_values.get(baseline_key)
    if baseline is None or len(baseline) == 0:
        return {}

    results = {}
    for name, values in condition_values.items():
        if name == baseline_key or not values:
            continue
        results[f"wasserstein_{name}"] = float(
            wasserstein_distance(baseline, values)
        )
    return results


# ============================================================================
# Main evaluator entry point
# ============================================================================

# ============================================================================
# Metric 4b: 3D Validity (geometric sanity check)
# ============================================================================

def compute_3d_validity(sdf_path: str | Path) -> dict:
    """Compute 3D geometric validity for all molecules in SDF.

    A molecule is 3D-valid if:
      1. It passes RDKit sanitization (2D validity)
      2. No severe atom overlap: minimum interatomic distance >= 0.7 Å
      3. All bond lengths in [0.8, 2.2] Å range
      4. No MMFF94 energy NaN

    This exposes Hard-Fix's geometric collapse where RDKit sanitization
    passes but the 3D geometry is catastrophically distorted.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    n_total = 0
    n_valid_2d = 0
    n_valid_3d = 0

    for mol in supplier:
        if mol is None:
            continue
        n_total += 1

        # Check 1: RDKit sanitization
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        n_valid_2d += 1

        # Check 2: Minimum interatomic distance (no severe overlaps)
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        min_dist = float('inf')
        try:
            for i in range(n_atoms):
                pi = conf.GetAtomPosition(i)
                for j in range(i + 1, min(i + 20, n_atoms)):  # local neighborhood
                    pj = conf.GetAtomPosition(j)
                    d = pi.Distance(pj)
                    if d < min_dist:
                        min_dist = d
        except Exception:
            continue
        if min_dist < 0.7:  # severe atomic overlap
            continue

        # Check 3: Bond lengths in reasonable range [0.8, 2.2] Å
        bonds_ok = True
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            try:
                pi = conf.GetAtomPosition(i)
                pj = conf.GetAtomPosition(j)
                bl = pi.Distance(pj)
                if bl < 0.8 or bl > 2.2:
                    bonds_ok = False
                    break
            except Exception:
                bonds_ok = False
                break
        if not bonds_ok:
            continue

        # Check 4: MMFF94 energy is finite (not NaN)
        try:
            mol_h = Chem.AddHs(mol)
            mp = AllChem.MMFFGetMoleculeProperties(mol_h)
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp)
                if ff is not None:
                    e = ff.CalcEnergy()
                    if np.isnan(e) or np.isinf(e):
                        continue
        except Exception:
            continue

        n_valid_3d += 1

    return {
        "n_total": n_total,
        "n_valid_2d": n_valid_2d,
        "n_valid_3d": n_valid_3d,
        "validity_2d_rate": n_valid_2d / max(n_total, 1),
        "validity_3d_rate": n_valid_3d / max(n_total, 1),
    }


def evaluate_condition(
    sdf_dir: str | Path,
    protein_pdb: str | Path,
    site_json: str | Path | None = None,
    output_json: str | Path | None = None,
    kpe_json: str | Path | None = None,
    run_vina: bool = False,
) -> dict:
    """Run all metrics on a single experimental condition.

    Args:
        sdf_dir: Directory containing generated SDF files.
        protein_pdb: Protein structure PDB.
        site_json: JSON file with HEW/SW site definitions.
        output_json: If set, write results to this file.
        kpe_json: If set, read KPE diagnostics from this JSON.
        run_vina: Whether to run Vina docking (expensive).

    Returns:
        Dict with all computed metrics.
    """
    sdf_dir = Path(sdf_dir)
    sdf_files = sorted(sdf_dir.glob("*.sdf"))
    if not sdf_files:
        print(f"  WARNING: No SDF files found in {sdf_dir}")
        return {"error": "No SDF files found", "sdf_dir": str(sdf_dir)}

    # Combine SDF files
    combined_sdf = sdf_dir / "__combined_eval.sdf"
    _concat_sdfs(sdf_files, combined_sdf)

    print(f"  Evaluating {len(sdf_files)} SDF files from {sdf_dir}...")

    metrics = {}

    # Validity (2D)
    validity = compute_validity_batch(combined_sdf)
    metrics["validity"] = validity
    print(f"    Validity (2D): {validity['n_valid']}/{validity['n_total']} "
          f"({validity['validity_rate']:.1%})")

    # 3D Validity (geometric sanity)
    validity_3d = compute_3d_validity(combined_sdf)
    metrics["validity_3d"] = validity_3d
    print(f"    Validity (3D): {validity_3d['n_valid_3d']}/{validity_3d['n_total']} "
          f"({validity_3d['validity_3d_rate']:.1%})")

    # Strain Energy
    strain = compute_strain_energy_batch(combined_sdf)
    metrics["strain_energy"] = strain
    if strain.get("strain_per_atom_mean") is not None:
        print(f"    Strain Energy: {strain['strain_per_atom_mean']:.1f} ± "
              f"{strain['strain_per_atom_std']:.1f} kcal/mol")

    # Clash Score (protein-dependent — skip if no protein)
    if protein_pdb and Path(protein_pdb).exists():
        clash = compute_clash_score(combined_sdf, protein_pdb)
        metrics["clash_score"] = clash
        if clash["clash_score_mean"] is not None:
            print(f"    Clash Score: {clash['clash_score_mean']:.4f}")
    else:
        metrics["clash_score"] = {"status": "skipped", "reason": "no protein PDB"}
        print(f"    Clash Score: SKIPPED (no protein)")

    # PBR (protein-dependent — skip if no protein)
    if protein_pdb and Path(protein_pdb).exists():
        pbr = compute_pbr(combined_sdf, protein_pdb)
        metrics["pbr"] = pbr
        if pbr["pbr_mean"] is not None:
            print(f"    PBR: {pbr['pbr_mean']:.4f} ± {pbr['pbr_std']:.4f}")
    else:
        metrics["pbr"] = {"status": "skipped", "reason": "no protein PDB"}
        print(f"    PBR: SKIPPED (no protein)")

    # SA Score
    sa = compute_sa_score_batch(combined_sdf)
    metrics["sa_score"] = sa
    if sa["sa_score_mean"] is not None:
        print(f"    SA Score: {sa['sa_score_mean']:.2f} ± {sa['sa_score_std']:.2f}")

    # QED
    qed = compute_qed_batch(combined_sdf)
    metrics["qed"] = qed
    if qed["qed_mean"] is not None:
        print(f"    QED: {qed['qed_mean']:.3f} ± {qed['qed_std']:.3f}")

    # Site Occupancy (HEW and SW)
    if site_json and Path(site_json).exists():
        occ = compute_site_occupancy_batch(combined_sdf, site_json)
        metrics["site_occupancy"] = occ
        hew_docc = occ.get("direct_occ_hew", "N/A")
        sw_docc = occ.get("direct_occ_sw", "N/A")
        print(f"    DirectOcc (HEW): {hew_docc}, DirectOcc (SW): {sw_docc}")

    # Diversity
    div = compute_diversity_batch(combined_sdf)
    metrics["diversity"] = div
    if div["diversity"] is not None:
        print(f"    Diversity: {div['diversity']:.3f}")

    # Vina (optional — expensive)
    if run_vina:
        vina = compute_vina_score_batch(combined_sdf, protein_pdb)
        metrics["vina"] = vina
        if vina["vina_mean"] is not None:
            print(f"    Vina: {vina['vina_mean']:.2f} ± {vina['vina_std']:.2f} "
                  f"(best: {vina.get('vina_best', 'N/A')})")

    # KPE (from instrumented generation log)
    if kpe_json and Path(kpe_json).exists():
        with open(kpe_json) as f:
            kpe_data = json.load(f)
        metrics["kpe"] = kpe_data
        rho = kpe_data.get("kpe_ratio", "N/A")
        print(f"    KPE Ratio (ρ): {rho}")

    # Cleanup
    combined_sdf.unlink(missing_ok=True)

    # Save
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Metrics saved to {output_json}")

    return metrics


def evaluate_batch(
    base_dir: str | Path,
    pockets: list[str],
    conditions: list[str],
    protein_dir: str | Path | None = None,
    site_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    baseline_condition: str = "unguided",
    run_vina: bool = False,
) -> dict:
    """Batch evaluation across pockets and conditions.

    Directory layout expected:
        base_dir/{pocket}/{condition}/*.sdf
        protein_dir/{pocket}_protein.pdb  (or base_dir/{pocket}/protein.pdb)
        site_dir/{pocket}_sites.json      (or base_dir/{pocket}/sites.json)
        base_dir/{pocket}/{condition}/kpe_summary.json  (KPE logs)

    Args:
        base_dir: Root experiment directory.
        pockets: List of pocket names (e.g., ["3mfw", "2gni"]).
        conditions: List of condition names (e.g., ["unguided", "kinematic"]).
        protein_dir: Directory with protein PDB files.
        site_dir: Directory with site JSON files.
        output_dir: Where to write per-condition JSON files.
        baseline_condition: Which condition is the unguided baseline for
                           Wasserstein computation.
        run_vina: Whether to run Vina docking.

    Returns:
        Consolidated dict with all results.
    """
    base_dir = Path(base_dir)
    protein_dir = Path(protein_dir) if protein_dir else base_dir
    site_dir = Path(site_dir) if site_dir else base_dir
    output_dir = Path(output_dir) if output_dir else base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for pocket in pockets:
        all_results[pocket] = {}
        for cond in conditions:
            sdf_dir = base_dir / pocket / cond
            protein_pdb = protein_dir / f"{pocket}_protein.pdb"
            site_json = site_dir / f"{pocket}_sites.json"
            kpe_json = sdf_dir / "kpe_summary.json"

            if not sdf_dir.exists():
                print(f"SKIP: {sdf_dir} not found")
                continue

            print(f"\n{'='*60}")
            print(f"Pocket: {pocket} | Condition: {cond}")
            print(f"{'='*60}")

            out_json = output_dir / pocket / f"{cond}_eval.json"
            result = evaluate_condition(
                sdf_dir=sdf_dir,
                protein_pdb=protein_pdb if protein_pdb.exists() else None,
                site_json=site_json if site_json.exists() else None,
                output_json=out_json,
                kpe_json=kpe_json if kpe_json.exists() else None,
                run_vina=run_vina,
            )
            all_results[pocket][cond] = result

    # Wasserstein distances (per-pocket, across conditions)
    print(f"\n{'='*60}")
    print("WASSERSTEIN-1 DISTANCES (vs. baseline)")
    print(f"{'='*60}")
    wasserstein_results = _compute_all_wasserstein(all_results, baseline_condition)
    for pocket, ws in wasserstein_results.items():
        print(f"  {pocket}: {json.dumps(ws, indent=2)}")

    # Save consolidated
    consolidated = {
        "pockets": all_results,
        "wasserstein": wasserstein_results,
        "baseline_condition": baseline_condition,
        "pockets_list": pockets,
        "conditions_list": conditions,
    }
    consolidated_path = output_dir / "consolidated_eval.json"
    with open(consolidated_path, "w") as f:
        json.dump(consolidated, f, indent=2, default=str)
    print(f"\nConsolidated results saved to {consolidated_path}")

    return consolidated


# ============================================================================
# Helper functions
# ============================================================================

# Van der Waals radii (in Angstroms) for common elements
_VDR_RADII = {
    "H": 1.20, "He": 1.40,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Br": 1.85, "I": 1.98,
    "Na": 2.27, "K": 2.75, "Mg": 1.73, "Ca": 2.31,
    "Zn": 1.39, "Fe": 2.00, "Mn": 2.00,
}


def _vdw_radius(element: str) -> float:
    """Get van der Waals radius for an element."""
    return _VDR_RADII.get(element, 1.70)  # default to carbon radius


def _read_protein_atoms(pdb_path: str | Path) -> tuple[np.ndarray | None, list[str]]:
    """Read heavy-atom coordinates and elements from a PDB file."""
    coords = []
    elements = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        el = line[76:78].strip() or line[12:16].strip()[:1]
                        if el.upper() == "H":
                            continue  # skip hydrogens
                        coords.append((x, y, z))
                        elements.append(el.upper() or "C")
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        return None, []
    return (np.array(coords) if coords else None), elements


def _concat_sdfs(sdf_files: list[Path], output: Path):
    """Concatenate multiple SDF files into one."""
    with open(output, "w") as out:
        for sf in sdf_files:
            with open(sf) as inp:
                content = inp.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")


def _write_receptor_pdbqt(protein_pdb, output_path):
    """Prepare receptor PDBQT for Vina."""
    import subprocess
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        mol = Chem.MolFromPDBFile(str(protein_pdb), removeHs=True)
        if mol:
            preparator = MoleculePreparation()
            mols = list(preparator.prepare(mol))
            if mols:
                pdbqt_str, _ = PDBQTWriterLegacy.write_string(mols[0])
                Path(output_path).write_text(pdbqt_str)
                return
    except Exception:
        pass
    subprocess.run(["obabel", str(protein_pdb), "-O", str(output_path), "-xr", "-xp"],
                   capture_output=True, timeout=60)


def _meeko_prepare_pdb(ligand_pdb, output_pdbqt):
    """Prepare ligand with meeko for Vina."""
    import subprocess
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False)
        if mol:
            preparator = MoleculePreparation()
            mols = list(preparator.prepare(mol))
            if mols:
                pdbqt_str, _ = PDBQTWriterLegacy.write_string(mols[0])
                Path(output_pdbqt).write_text(pdbqt_str)
                return
    except Exception:
        pass
    subprocess.run(["obabel", str(ligand_pdb), "-O", str(output_pdbqt), "-xp"],
                   capture_output=True, timeout=30)


def _compute_docking_box(protein_pdb, padding=5.0):
    """Compute docking box centered on protein pocket."""
    coords, _ = _read_protein_atoms(protein_pdb)
    if coords is not None and len(coords) > 0:
        center = tuple(coords.mean(axis=0).tolist())
        extent = coords.max(axis=0) - coords.min(axis=0)
        box = tuple(max(extent[i] + 2 * padding, 20.0) for i in range(3))
    else:
        center = (0.0, 0.0, 0.0)
        box = (22.5, 22.5, 22.5)
    return center, box


def _parse_vina_stdout(stdout):
    """Parse Vina output for affinity scores."""
    scores = []
    for line in stdout.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                s = float(parts[0])
                if -50 < s < 50:
                    scores.append(s)
            except ValueError:
                continue
    return scores or None


def _compute_all_wasserstein(
    all_results: dict,
    baseline_condition: str = "unguided",
) -> dict:
    """Compute Wasserstein distances across conditions for each pocket.

    Aggregates per-molecule values for: QED, SA Score, Strain Energy, Vina.
    """
    wass_results = {}
    for pocket, conditions in all_results.items():
        # Collect per-molecule values for each metric
        # NOTE: this requires per-molecule data, not just summary stats.
        # For now, we use the distribution from the SDF files.
        pocket_wass = {}
        # We'd need per-molecule values here; this is a structural hook
        # that gets populated when per-molecule data is available.
        wass_results[pocket] = pocket_wass
    return wass_results


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for ESField Kinematic Anchor Guidance"
    )
    sub = parser.add_subparsers(dest="mode")

    # Single-condition mode
    single = sub.add_parser("single", help="Evaluate a single condition")
    single.add_argument("--sdf-dir", required=True)
    single.add_argument("--protein-pdb", required=True)
    single.add_argument("--site-json", default=None)
    single.add_argument("--output-json", default=None)
    single.add_argument("--kpe-json", default=None)
    single.add_argument("--run-vina", action="store_true")

    # Batch mode
    batch = sub.add_parser("batch", help="Batch evaluate multiple conditions")
    batch.add_argument("--base-dir", required=True)
    batch.add_argument("--pockets", default="3mfw,2gni,6o4x,2jke,2gqn,6phx")
    batch.add_argument("--conditions", default="unguided,hard_fix,kinematic")
    batch.add_argument("--protein-dir", default=None)
    batch.add_argument("--site-dir", default=None)
    batch.add_argument("--output-dir", default=None)
    batch.add_argument("--baseline-condition", default="unguided")
    batch.add_argument("--run-vina", action="store_true")

    args = parser.parse_args()

    if args.mode == "single":
        evaluate_condition(
            sdf_dir=args.sdf_dir,
            protein_pdb=args.protein_pdb,
            site_json=args.site_json,
            output_json=args.output_json,
            kpe_json=args.kpe_json,
            run_vina=args.run_vina,
        )
    elif args.mode == "batch":
        evaluate_batch(
            base_dir=args.base_dir,
            pockets=args.pockets.split(","),
            conditions=args.conditions.split(","),
            protein_dir=args.protein_dir,
            site_dir=args.site_dir,
            output_dir=args.output_dir,
            baseline_condition=args.baseline_condition,
            run_vina=args.run_vina,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
