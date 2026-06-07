"""Site occupancy evaluation metrics for v7 two-stage generation.

Provides lightweight metrics to assess whether generated molecules
actually occupy candidate HEW sites with compatible atom types.

Key metrics:
  - direct_occupancy_rate: fraction of molecules that occupy ≥1 site
  - best_compatible_distance: nearest compatible-atom distance per site
  - site_occupancy_summary: combined diagnostic report

These metrics are specifically designed to evaluate the v7 two-stage
generation approach, complementing the existing POSU-v2.1/HEWU metrics.

Dependencies: RDKit (pip install rdkit)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rdkit import Chem


# ---------------------------------------------------------------------------
# Atom type compatibility (mirrors src/guidance/latent_guidance.py)
# ---------------------------------------------------------------------------

HEW_ENV_HYDROPHOBIC = "hydrophobic"
HEW_ENV_POLAR_UNSATISFIED = "polar_unsatisfied"
HEW_ENV_MIXED = "mixed"
HEW_ENV_BURIED = "buried"

# Hard-coded compatibility: which atom types are compatible with each
# HEW environment.  Mirrors the latent_guidance compatibility matrix but
# uses RDKit-based atom classification.
_HEW_COMPATIBLE_ELEMENTS: dict[str, set[int]] = {
    HEW_ENV_HYDROPHOBIC: {6, 9, 16, 17, 35, 53},  # C, F, S, Cl, Br, I
    HEW_ENV_POLAR_UNSATISFIED: {7, 8},               # N, O
    HEW_ENV_MIXED: {6, 7, 8, 9, 16, 17, 35, 53},    # C, N, O, F, S, halogens
    HEW_ENV_BURIED: {6, 9, 17, 35, 53},              # C, F, Cl, Br, I (small)
}

# Compatibility scores for types within each environment (for weighted metrics)
_HEW_COMPAT_SCORES: dict[str, dict[int, float]] = {
    HEW_ENV_HYDROPHOBIC: {6: 1.0, 9: 1.0, 16: 0.3, 17: 1.0, 35: 1.0, 53: 1.0, 7: -0.5, 8: -0.5},
    HEW_ENV_POLAR_UNSATISFIED: {7: 1.0, 8: 1.0, 16: 0.3, 6: -0.3, 9: -0.3, 17: -0.3, 35: -0.3, 53: -0.3},
    HEW_ENV_MIXED: {6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5, 16: 0.5, 17: 0.5, 35: 0.5, 53: 0.5},
    HEW_ENV_BURIED: {6: 0.3, 9: 0.5, 17: 0.5, 35: 0.5, 53: 0.5, 7: -1.0, 8: -1.0},
}


def classify_hew_environment(site: dict) -> str:
    """Classify a HEW site by local environment (same logic as posu.py)."""
    features = site.get("features", {})
    hbond = features.get("hbond_count", 0)
    hydrophobic = features.get("hydrophobic_contact_count", 0)
    nearest_dist = features.get("nearest_protein_distance", 4.0)

    if nearest_dist < 2.5:
        return HEW_ENV_BURIED
    if hydrophobic >= 4 and hbond <= 1:
        return HEW_ENV_HYDROPHOBIC
    if hbond <= 1 and hydrophobic <= 2:
        return HEW_ENV_POLAR_UNSATISFIED
    return HEW_ENV_MIXED


def _get_atom_info(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Extract atom positions and elements from RDKit mol.

    Returns list of dicts with keys: coord (3-tuple), atomic_num (int).
    Hydrogens are excluded.
    """
    conf = mol.GetConformer()
    atoms = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(a.GetIdx())
        atoms.append({
            "coord": (pos.x, pos.y, pos.z),
            "atomic_num": a.GetAtomicNum(),
            "idx": a.GetIdx(),
        })
    return atoms


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Primary metrics
# ---------------------------------------------------------------------------


def direct_occupancy_rate(
    generated_mols: list,
    site_map: dict[str, Any] | str | Path,
    *,
    threshold: float = 2.5,
) -> dict[str, Any]:
    """Compute the fraction of molecules that occupy at least one HEW site.

    "Occupancy" means: at least one atom of a COMPATIBLE type is within
    `threshold` Å of a candidate HEW site center.

    Args:
        generated_mols: list of RDKit Mol objects, or path to SDF file
        site_map: site map dict or path to JSON
        threshold: distance threshold in Å (default 2.5)

    Returns:
        dict with:
          - "occupancy_rate": fraction of mols that occupy ≥1 site
          - "n_occupied": count of occupying mols
          - "n_total": total number of mols
          - "per_mol_occupied_sites": list of occupied-site counts per mol
          - "per_mol_best_distance": best compat distance per mol
    """
    # Load site map
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    # Load molecules
    if isinstance(generated_mols, (str, Path)):
        generated_mols = list(
            Chem.SDMolSupplier(str(generated_mols), sanitize=False)
        )

    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return {
            "occupancy_rate": 0.0,
            "n_occupied": 0,
            "n_total": len(generated_mols),
            "per_mol_occupied_sites": [],
            "per_mol_best_distance": [],
            "message": "No HEW sites in site map",
        }

    n_occupied = 0
    per_mol_occupied = []
    per_mol_best_dist = []

    for mol in generated_mols:
        if mol is None:
            per_mol_occupied.append(0)
            per_mol_best_dist.append(float("inf"))
            continue

        try:
            atoms = _get_atom_info(mol)
        except Exception:
            per_mol_occupied.append(0)
            per_mol_best_dist.append(float("inf"))
            continue

        mol_occupied = 0
        mol_best_dist = float("inf")

        for site in hew_sites:
            env = classify_hew_environment(site)
            center = tuple(site["center"])
            compat_elements = _HEW_COMPATIBLE_ELEMENTS.get(env, set())

            best_compat_dist = float("inf")
            for atom in atoms:
                d = _distance(atom["coord"], center)
                if atom["atomic_num"] in compat_elements:
                    if d < best_compat_dist:
                        best_compat_dist = d

            if best_compat_dist <= threshold:
                mol_occupied += 1

            if best_compat_dist < mol_best_dist:
                mol_best_dist = best_compat_dist

        if mol_occupied > 0:
            n_occupied += 1

        per_mol_occupied.append(mol_occupied)
        per_mol_best_dist.append(mol_best_dist)

    n_total = len(generated_mols)
    return {
        "occupancy_rate": n_occupied / n_total if n_total > 0 else 0.0,
        "n_occupied": n_occupied,
        "n_total": n_total,
        "per_mol_occupied_sites": per_mol_occupied,
        "per_mol_best_distance": per_mol_best_dist,
    }


def best_compatible_distance(
    generated_mols: list,
    site_map: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute the nearest compatible-atom distance for each HEW site.

    For each candidate HEW site, finds the minimum distance among all
    atoms whose element is compatible with the site's environment.

    Args:
        generated_mols: list of RDKit Mol objects, or path to SDF
        site_map: site map dict or path to JSON

    Returns:
        dict with:
          - "per_site_best_distance": list of (site_idx, env, distance, atomic_num)
          - "mean_best_distance": average across all sites
          - "min_best_distance": overall nearest compatible atom
          - "n_sites_occupied": sites with best distance ≤ 2.5 Å
          - "n_sites_total": total HEW sites
    """
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    if isinstance(generated_mols, (str, Path)):
        generated_mols = list(
            Chem.SDMolSupplier(str(generated_mols), sanitize=False)
        )

    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return {
            "per_site_best_distance": [],
            "mean_best_distance": float("inf"),
            "min_best_distance": float("inf"),
            "n_sites_occupied": 0,
            "n_sites_total": 0,
        }

    # Collect all atoms from all molecules
    all_atoms = []
    for mol in generated_mols:
        if mol is None:
            continue
        try:
            all_atoms.extend(_get_atom_info(mol))
        except Exception:
            continue

    per_site = []
    occupied_count = 0

    for site in hew_sites:
        env = classify_hew_environment(site)
        center = tuple(site["center"])
        compat_elements = _HEW_COMPATIBLE_ELEMENTS.get(env, set())

        best_d = float("inf")
        best_atomic_num = 0

        for atom in all_atoms:
            if atom["atomic_num"] in compat_elements:
                d = _distance(atom["coord"], center)
                if d < best_d:
                    best_d = d
                    best_atomic_num = atom["atomic_num"]

        per_site.append({
            "site_id": site.get("site_id", -1),
            "environment": env,
            "best_distance": best_d,
            "best_atomic_num": best_atomic_num,
        })

        if best_d <= 2.5:
            occupied_count += 1

    distances = [s["best_distance"] for s in per_site]
    finite_distances = [d for d in distances if d < float("inf")]

    return {
        "per_site_best_distance": per_site,
        "mean_best_distance": (
            sum(finite_distances) / len(finite_distances)
            if finite_distances
            else float("inf")
        ),
        "min_best_distance": min(finite_distances) if finite_distances else float("inf"),
        "n_sites_occupied": occupied_count,
        "n_sites_total": len(hew_sites),
    }


def site_occupancy_summary(
    generated_mols: list,
    site_map: dict[str, Any] | str | Path,
    *,
    threshold: float = 2.5,
) -> dict[str, Any]:
    """Combined site occupancy diagnostic report.

    Merges direct_occupancy_rate and best_compatible_distance into a
    single summary dict suitable for logging and comparison across
    experimental conditions.

    Args:
        generated_mols: list of RDKit Mol objects or SDF path
        site_map: site map dict or path to JSON
        threshold: occupancy distance threshold

    Returns:
        Comprehensive dict with occupancy stats and per-site details.
    """
    occ = direct_occupancy_rate(generated_mols, site_map, threshold=threshold)
    bcd = best_compatible_distance(generated_mols, site_map)

    return {
        "direct_occupancy": {
            "rate": occ["occupancy_rate"],
            "n_occupied": occ["n_occupied"],
            "n_total": occ["n_total"],
        },
        "compatible_distance": {
            "mean": bcd["mean_best_distance"],
            "min": bcd["min_best_distance"],
            "n_sites_occupied": bcd["n_sites_occupied"],
            "n_sites_total": bcd["n_sites_total"],
        },
        "per_mol_occupied_sites": occ["per_mol_occupied_sites"],
        "per_mol_best_distance": occ["per_mol_best_distance"],
        "per_site_details": bcd["per_site_best_distance"],
    }


# ---------------------------------------------------------------------------
# Convenience: compute from SDF + site map paths
# ---------------------------------------------------------------------------


def evaluate_sdf_occupancy(
    sdf_path: str | Path,
    site_map_path: str | Path,
    *,
    threshold: float = 2.5,
) -> dict[str, Any]:
    """Evaluate site occupancy metrics for an SDF file.

    Args:
        sdf_path: path to SDF file with generated molecules
        site_map_path: path to site map JSON
        threshold: distance threshold for occupancy

    Returns:
        Full site_occupancy_summary dict.
    """
    mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False))
    site_map = json.loads(Path(site_map_path).read_text())
    return site_occupancy_summary(mols, site_map, threshold=threshold)
