"""POSU v2 — Physical Opportunity Site Utilization with environment-aware scoring.

Key changes from v1:
  - HEW compatible rules tightened by local environment classification
  - SW scoring split into 4 components: SWP, SWBR, SWCR, SWDP
  - Composite POSU uses native-ligand-calibrated weighting
  - SW no longer naively penalizes all water displacement

All metrics use ONLY geometric distance + hard-coded chemistry rules + site metadata.
ZERO learned parameters. Fully reproducible from SDF + site map.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rdkit import Chem

from utils.chemistry import (
    infer_atom_type,
    atomic_number,
    is_compatible_atom_site,
    normalize_element,
    HYDROPHOBIC_ELEMENTS,
    POLAR_ELEMENTS,
)
from utils.geometry import distance as calc_distance

# ---------------------------------------------------------------------------
# Atom-level utilities (unchanged from v1)
# ---------------------------------------------------------------------------

def _extract_atoms_from_mol(mol: Chem.Mol) -> list[dict[str, Any]]:
    conf = mol.GetConformer()
    atoms = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(a.GetIdx())
        elem = a.GetSymbol()
        atoms.append({
            "coord": (pos.x, pos.y, pos.z),
            "atom_type": infer_atom_type(elem),
            "atomic_number": atomic_number(elem),
            "element": normalize_element(elem),
            "idx": a.GetIdx(),
        })
    return atoms


def _gaussian_proximity(distance: float, sigma: float) -> float:
    if distance > 5 * sigma:
        return 0.0
    return math.exp(-(distance**2) / (2 * sigma**2))


def _molecular_integrity_factor(atoms: list[dict], contact_cutoff: float = 3.0) -> float:
    if len(atoms) <= 1:
        return 1.0
    n_with_neighbor = 0
    for i, a1 in enumerate(atoms):
        for j, a2 in enumerate(atoms):
            if i >= j: continue
            d = calc_distance(a1["coord"], a2["coord"])
            if d <= contact_cutoff:
                n_with_neighbor += 1
                break
    fraction_connected = n_with_neighbor / len(atoms)
    return fraction_connected ** 2


def _is_hydrophobic_atom(atom_type: str, element: str) -> bool:
    if element in HYDROPHOBIC_ELEMENTS:
        return True
    return atom_type in {"C_sp3", "C_aromatic", "halogen", "S"}


def _is_strongly_polar(atom_type: str, element: str) -> bool:
    if element in {"N", "O"} and atom_type in {"O_acceptor", "N_donor", "N_acceptor"}:
        return True
    return False


def _is_hbond_capable(atom_type: str) -> bool:
    """Atoms capable of forming H-bonds (donor or acceptor)."""
    return atom_type in {"O_acceptor", "N_donor", "N_acceptor"}


# ---------------------------------------------------------------------------
# HEW environment classification (v2)
# ---------------------------------------------------------------------------

HEW_ENV_HYDROPHOBIC = "hydrophobic"
HEW_ENV_POLAR_UNSATISFIED = "polar_unsatisfied"
HEW_ENV_MIXED = "mixed"
HEW_ENV_BURIED = "buried"


def classify_hew_environment(site: dict) -> str:
    """Classify HEW site by local environment using site features.

    Uses hbond_count and hydrophobic_contact_count from crystal water analysis.

    Classification rules:
      - buried: nearest_protein_distance < 2.5Å or very isolated
      - hydrophobic: hydrophobic_contact_count >= 4 and hbond_count <= 1
      - polar_unsatisfied: hbond_count <= 1 and hydrophobic_contact_count <= 2
      - mixed: everything else
    """
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


# Tightened compatible atom types per HEW environment
HEW_COMPATIBLE_V2 = {
    HEW_ENV_HYDROPHOBIC: frozenset({"C_sp3", "C_aromatic", "halogen", "S"}),
    HEW_ENV_POLAR_UNSATISFIED: frozenset({"O_acceptor", "N_donor", "N_acceptor"}),
    HEW_ENV_MIXED: frozenset({"C_sp3", "C_aromatic", "halogen", "S", "O_acceptor", "N_donor", "N_acceptor"}),
    HEW_ENV_BURIED: frozenset({"C_sp3", "halogen"}),
}


def is_compatible_hew_v2(atom_type: str, atomic_number_value: int, site: dict) -> bool:
    """Check if an atom is compatible with a HEW site, considering environment."""
    env = classify_hew_environment(site)
    allowed = HEW_COMPATIBLE_V2.get(env, frozenset())
    if atom_type in allowed:
        return True
    # Fallback: atomic number check for the environment
    if env == HEW_ENV_HYDROPHOBIC and atomic_number_value in {6, 9, 16, 17, 35, 53}:
        return True
    if env in (HEW_ENV_POLAR_UNSATISFIED, HEW_ENV_MIXED) and atomic_number_value in {7, 8}:
        return True
    return False


# ---------------------------------------------------------------------------
# HEWU v2: environment-aware high-energy water replacement utility
# ---------------------------------------------------------------------------

def compute_hewu(
    molecule: Chem.Mol,
    site_map: dict[str, Any] | str | Path,
    *,
    sigma_scale: float = 1.5,
) -> dict[str, float]:
    """HEWU v2 — environment-aware HEW replacement utility.

    For each HEW site, compatible atoms are determined by local environment.
    Uses top-2 mean to dilute single-atom "lucky hits".
    """
    atoms = _extract_atoms_from_mol(molecule)
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return {"per_site": [], "mean_utility": 0.0, "total_utility": 0.0, "n_sites": 0}

    per_site = []
    env_counts = {}
    for site in hew_sites:
        env = classify_hew_environment(site)
        env_counts[env] = env_counts.get(env, 0) + 1
        sigma = max(0.1, site["radius"] * sigma_scale)
        utilities = []
        for atom in atoms:
            d = calc_distance(atom["coord"], tuple(site["center"]))
            prox = _gaussian_proximity(d, sigma)
            compat = 1.0 if is_compatible_hew_v2(atom["atom_type"], atom["atomic_number"], site) else 0.0
            utilities.append(prox * compat * site.get("confidence", 1.0))
        utilities.sort(reverse=True)
        top_k = 2
        top_vals = [v for v in utilities[:top_k] if v > 0]
        utility = sum(top_vals) / len(top_vals) if top_vals else 0.0
        # v2.1: downweight mixed-environment HEW (too broad compatibility)
        if env == HEW_ENV_MIXED:
            utility *= 0.6
        per_site.append(utility)

    return {
        "per_site": per_site,
        "mean_utility": sum(per_site) / len(per_site) if per_site else 0.0,
        "total_utility": sum(per_site),
        "n_sites": len(per_site),
        "env_counts": env_counts,
    }


# ---------------------------------------------------------------------------
# SW scoring v2: 4-component stable water handling
# ---------------------------------------------------------------------------

SW_BRIDGE_DISTANCE = 3.2  # ideal water-bridged H-bond distance (water O to ligand polar atom)
SW_DIRECT_DISTANCE = 3.0   # direct protein-ligand H-bond (compensated replacement)
SW_DISRUPTION_DISTANCE = 2.5  # distance within which an incompatible atom is destructive


def _compute_sw_score_per_site(
    atoms: list[dict],
    site: dict,
    sigma_scale: float,
) -> dict[str, float]:
    """Compute all 4 SW sub-scores for one stable water site.

    SWP  (preservation):      no incompatible atoms within disruption range
    SWBR (bridge reward):     polar atom at ideal bridge distance → water-mediated interaction
    SWCR (compensated):       polar atom directly occupies water site → compensates displacement
    SWDP (destructive):       incompatible atom near water AND no compensation → penalty

    Returns dict with all sub-scores and the combined SW score.
    """
    sigma = max(0.1, site["radius"] * sigma_scale)
    conf = site.get("confidence", 1.0)

    best_preservation = 1.0
    best_bridge = 0.0
    best_compensated = 0.0
    worst_destructive = 0.0
    has_compensation = False
    has_disruption = False

    for atom in atoms:
        d = calc_distance(atom["coord"], tuple(site["center"]))
        is_compat = is_compatible_atom_site(atom["atom_type"], atom["atomic_number"], "stable_water")
        is_polar = _is_hbond_capable(atom["atom_type"])

        # SWP: measure how "safe" the site is (no incompatible atoms nearby)
        # v2.1: hard distance threshold — only count as disruption if within 2.5Å
        if not is_compat:
            if d < SW_DISRUPTION_DISTANCE:
                prox = _gaussian_proximity(d, sigma)
                disruption = prox * conf
                has_disruption = True
                if (1.0 - disruption) < best_preservation:
                    best_preservation = 1.0 - disruption

        # SWBR: polar atom at ideal bridge distance (water-mediated H-bond)
        if is_polar:
            bridge_prox = _gaussian_proximity(abs(d - SW_BRIDGE_DISTANCE), 0.8 * sigma_scale)
            bridge_score = bridge_prox * conf
            if bridge_score > best_bridge:
                best_bridge = bridge_score

        # SWCR: polar atom at direct interaction distance (compensated replacement)
        if is_compat and is_polar:
            direct_prox = _gaussian_proximity(d, sigma)
            comp_score = direct_prox * conf
            if comp_score > best_compensated:
                best_compensated = comp_score
            if comp_score > 0.1:
                has_compensation = True

        # SWDP: incompatible atom that displaces water WITHOUT compensation
        if not is_compat and not is_polar:
            prox = _gaussian_proximity(d, sigma)
            destructive = prox * conf
            if destructive > worst_destructive:
                worst_destructive = destructive

    # SW total logic:
    # If no disruption at all → full preservation score
    # If disruption but compensated (polar atom replaces water) → acceptable, mild penalty
    # If disruption without compensation → penalty
    if not has_disruption:
        sw_total = 1.0  # perfect preservation
    elif has_compensation:
        # Compensated replacement: penalize residual disruption but reward compensation
        sw_total = max(0.0, 1.0 - 0.5 * worst_destructive + 0.3 * best_compensated)
        sw_total = min(1.0, sw_total)
    else:
        # Destructive displacement: full penalty
        sw_total = max(0.0, 1.0 - worst_destructive - 0.1 * best_bridge)

    return {
        "swp": best_preservation,
        "swbr": best_bridge,
        "swcr": best_compensated,
        "swdp_penalty": worst_destructive,
        "has_disruption": has_disruption,
        "has_compensation": has_compensation,
        "sw_total": sw_total,
    }


def compute_sw_score(
    molecule: Chem.Mol,
    site_map: dict[str, Any] | str | Path,
    *,
    sigma_scale: float = 1.5,
) -> dict[str, Any]:
    """SW Score v2 — multi-component stable water handling.

    Replaces the single SWDP penalty from v1.

    Returns:
        dict with per-site sub-scores and aggregate
    """
    atoms = _extract_atoms_from_mol(molecule)
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())
    sw_sites = [s for s in site_map["sites"] if s["site_type"] == "stable_water"]
    if not sw_sites:
        return {
            "per_site": [], "mean_swp": 1.0, "mean_swbr": 0.0,
            "mean_swcr": 0.0, "mean_swdp": 0.0, "mean_sw_total": 1.0,
            "n_sites": 0, "n_disrupted": 0, "n_compensated": 0,
        }

    per_site = []
    swp_vals, swbr_vals, swcr_vals, swdp_vals, sw_total_vals = [], [], [], [], []
    n_disrupted = 0
    n_compensated = 0

    for site in sw_sites:
        scores = _compute_sw_score_per_site(atoms, site, sigma_scale)
        per_site.append(scores)
        swp_vals.append(scores["swp"])
        swbr_vals.append(scores["swbr"])
        swcr_vals.append(scores["swcr"])
        swdp_vals.append(scores["swdp_penalty"])
        sw_total_vals.append(scores["sw_total"])
        if scores["has_disruption"]:
            n_disrupted += 1
        if scores["has_compensation"]:
            n_compensated += 1

    return {
        "per_site": per_site,
        "mean_swp": sum(swp_vals) / len(swp_vals),
        "mean_swbr": sum(swbr_vals) / len(swbr_vals),
        "mean_swcr": sum(swcr_vals) / len(swcr_vals),
        "mean_swdp": sum(swdp_vals) / len(swdp_vals),
        "mean_sw_total": sum(sw_total_vals) / len(sw_total_vals),
        "n_sites": len(sw_sites),
        "n_disrupted": n_disrupted,
        "n_compensated": n_compensated,
    }


# Backward-compatible alias
def compute_swdp(
    molecule: Chem.Mol,
    site_map: dict[str, Any] | str | Path,
    *,
    sigma_scale: float = 1.5,
) -> dict[str, Any]:
    """Deprecated v1 alias — returns v2 SW score with v1-compatible keys."""
    result = compute_sw_score(molecule, site_map, sigma_scale=sigma_scale)
    return {
        "per_site": [s["swdp_penalty"] for s in result["per_site"]],
        "mean_penalty": result["mean_swdp"],
        "total_penalty": result["mean_swdp"] * result["n_sites"],
        "n_sites": result["n_sites"],
    }


# ---------------------------------------------------------------------------
# HCFU v2: hydrophobic cavity filling (strengthened polar penalty)
# ---------------------------------------------------------------------------

def compute_hcfu(
    molecule: Chem.Mol,
    site_map: dict[str, Any] | str | Path,
    *,
    sigma_scale: float = 1.5,
) -> dict[str, float]:
    """HCFU v2 — hydrophobic cavity filling with stronger polar penalty.

    For each HC site:
      max_i [ prox(i,j) * hydrophobic(i) ] - 2.0 * max_i [ prox(i,j) * polar(i) ]
    Higher = better filling with hydrophobic atoms, stronger penalty for polar intrusion.
    """
    atoms = _extract_atoms_from_mol(molecule)
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())
    hc_sites = [s for s in site_map["sites"] if s["site_type"] == "hydrophobic_cavity"]
    if not hc_sites:
        return {"per_site": [], "mean_utility": 0.0, "total_utility": 0.0, "n_sites": 0}

    per_site = []
    for site in hc_sites:
        sigma = max(0.1, site["radius"] * sigma_scale)
        hydrophobic_scores = []
        polar_scores = []
        for atom in atoms:
            d = calc_distance(atom["coord"], tuple(site["center"]))
            prox = _gaussian_proximity(d, sigma)
            if _is_hydrophobic_atom(atom["atom_type"], atom["element"]):
                hydrophobic_scores.append(prox)
            else:
                hydrophobic_scores.append(0.0)
            if _is_strongly_polar(atom["atom_type"], atom["element"]):
                polar_scores.append(prox * site.get("confidence", 1.0))
            else:
                polar_scores.append(0.0)
        hydrophobic_scores.sort(reverse=True)
        polar_scores.sort(reverse=True)
        top_k = 2
        top_h = [v for v in hydrophobic_scores[:top_k] if v > 0]
        top_p = [v for v in polar_scores[:top_k] if v > 0]
        best_hydrophobic = sum(top_h) / len(top_h) if top_h else 0.0
        worst_polar = sum(top_p) / len(top_p) if top_p else 0.0
        # v2: double the polar penalty weight
        utility = max(0.0, best_hydrophobic - 2.0 * worst_polar) * site.get("confidence", 1.0)
        per_site.append(utility)

    return {
        "per_site": per_site,
        "mean_utility": sum(per_site) / len(per_site) if per_site else 0.0,
        "total_utility": sum(per_site),
        "n_sites": len(per_site),
    }


# ---------------------------------------------------------------------------
# POSU v2 composite
# ---------------------------------------------------------------------------

def compute_posu(
    molecule: Chem.Mol,
    site_map: dict[str, Any] | str | Path,
    *,
    sigma_scale: float = 1.5,
    hew_weight: float = 1.2,   # HEW is the primary optimization target
    sw_weight: float = 0.6,    # SW is a constraint, not an optimization target
    hc_weight: float = 0.8,    # HC is secondary
    apply_integrity_check: bool = True,
) -> dict[str, Any]:
    """POSU v2 — Physical Opportunity Site Utilization.

    Key changes from v1:
      - HEWU: environment-aware compatible rules
      - SW: 4-component scoring (preservation/bridge/compensated/destructive)
      - HCFU: strengthened polar penalty (2x)
      - Weighting: HEW > HC > SW (calibrated via native ligand)
      - SW no longer cancels HEW improvement in composite

    POSU = (w_HEW * HEWU + w_SW * SW_total + w_HC * HCFU) / sum(weights)
    """
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    all_atoms = _extract_atoms_from_mol(molecule)
    integrity = _molecular_integrity_factor(all_atoms) if apply_integrity_check else 1.0

    hewu = compute_hewu(molecule, site_map, sigma_scale=sigma_scale)
    sw_score = compute_sw_score(molecule, site_map, sigma_scale=sigma_scale)
    hcfu = compute_hcfu(molecule, site_map, sigma_scale=sigma_scale)

    hew_mean = hewu["mean_utility"]
    sw_mean = sw_score["mean_sw_total"]
    hc_mean = hcfu["mean_utility"]

    weights = []
    terms = []
    if hewu["n_sites"] > 0:
        weights.append(hew_weight)
        terms.append(hew_mean)
    if sw_score["n_sites"] > 0:
        weights.append(sw_weight)
        terms.append(sw_mean)
    if hcfu["n_sites"] > 0:
        weights.append(hc_weight)
        terms.append(hc_mean)

    if not weights:
        posu = 0.0
    else:
        posu = sum(w * t for w, t in zip(weights, terms)) / sum(weights)

    posu = posu * integrity

    return {
        "posu": posu,
        "integrity_factor": integrity,
        "hewu": hewu,
        "sw_score": sw_score,
        "hcfu": hcfu,
        "hew_mean": hew_mean,
        "sw_total": sw_mean,
        "hc_mean": hc_mean,
        "hew_env_counts": hewu.get("env_counts", {}),
        "sw_n_disrupted": sw_score["n_disrupted"],
        "sw_n_compensated": sw_score["n_compensated"],
    }


def compute_all_site_metrics(
    molecule: Chem.Mol,
    site_map: dict[str, Any] | str | Path,
    *,
    sigma_scale: float = 1.5,
) -> dict[str, Any]:
    """Compute all site metrics for one molecule: POSU v2 + sub-scores."""
    return compute_posu(molecule, site_map, sigma_scale=sigma_scale)
