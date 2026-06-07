"""Quality-constrained evaluation: Q-POSU.

Combines POSU with quality penalties to answer:
  "Did you improve site utilization by sacrificing molecular quality?"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, AllChem

from evaluation.posu import compute_posu


def _compute_tanimoto_duplicates(mols: Sequence[Chem.Mol]) -> float:
    """Estimate mode collapse via Tanimoto similarity of ECFP4 fingerprints."""
    from rdkit import DataStructs
    fps = []
    for m in mols:
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
        except Exception:
            continue
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    # Collapse rate = proportion of pairs with similarity > 0.95
    return sum(1 for s in sims if s > 0.95) / len(sims) if sims else 0.0


def compute_quality_penalty(
    molecules: Sequence[Chem.Mol],
    reference_molecules: Sequence[Chem.Mol] | None = None,
    *,
    vina_scores: Sequence[float] | None = None,
    vina_ref: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute quality penalty for a set of generated molecules.

    Quality penalty components (each 0 to 1, averaged):
      - QED_drop: relative drop vs reference mean QED
      - validity_loss: proportion of invalid molecules
      - collapse_rate: proportion of near-duplicate pairs (Tanimoto > 0.95)
      - MW_inflation: relative MW increase vs reference
      - Vina_drop: relative Vina worsening vs reference

    If no reference provided, uses absolute thresholds.

    Returns:
        dict with per-component and total penalty
    """
    n_total = len(molecules)
    valid_mols = []
    for m in molecules:
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            valid_mols.append(m)
        except Exception:
            pass

    n_valid = len(valid_mols)
    validity_loss = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0

    # QED
    qeds = []
    mws = []
    for m in valid_mols:
        try:
            qeds.append(QED.qed(m))
            mws.append(Descriptors.MolWt(m))
        except Exception:
            pass

    qed_mean = float(np.mean(qeds)) if qeds else 0.0
    mw_mean = float(np.mean(mws)) if mws else 0.0

    # Reference stats
    if reference_molecules:
        ref_valid = []
        for m in reference_molecules:
            if m is None:
                continue
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                ref_valid.append(m)
            except Exception:
                pass
        ref_qeds = [QED.qed(m) for m in ref_valid] if ref_valid else [0.5]
        ref_mws = [Descriptors.MolWt(m) for m in ref_valid] if ref_valid else [400]
        ref_qed_mean = float(np.mean(ref_qeds))
        ref_mw_mean = float(np.mean(ref_mws))
        ref_vina_mean = float(np.mean([v for v in vina_ref if v is not None])) if vina_ref else None
    else:
        ref_qed_mean = 0.5  # absolute reference
        ref_mw_mean = 400.0
        ref_vina_mean = None

    # QED drop penalty (only if worse)
    qed_drop = max(0.0, ref_qed_mean - qed_mean) / max(0.1, ref_qed_mean)

    # MW inflation penalty (only if significantly larger)
    mw_inflation = max(0.0, (mw_mean - ref_mw_mean) / max(1.0, ref_mw_mean))

    # Collapse rate
    collapse_rate = _compute_tanimoto_duplicates(valid_mols)

    # Vina drop
    vina_drop = 0.0
    if vina_scores and ref_vina_mean is not None:
        vina_valid = [v for v in vina_scores if v is not None]
        vina_mean = float(np.mean(vina_valid)) if vina_valid else 0.0
        vina_drop = max(0.0, vina_mean - ref_vina_mean) / max(0.1, abs(ref_vina_mean))

    # Total penalty (equal weight)
    components = {
        "validity_loss": validity_loss,
        "qed_drop": qed_drop,
        "mw_inflation": mw_inflation,
        "collapse_rate": collapse_rate,
        "vina_drop": vina_drop,
    }
    total_penalty = float(np.mean(list(components.values())))

    return {
        "total_penalty": total_penalty,
        "components": components,
        "qed_mean": qed_mean,
        "mw_mean": mw_mean,
        "n_valid": n_valid,
        "n_total": n_total,
    }


def compute_q_posu(
    molecules: Sequence[Chem.Mol],
    site_map: dict[str, Any] | str | Path,
    reference_molecules: Sequence[Chem.Mol] | None = None,
    *,
    vina_scores: Sequence[float] | None = None,
    vina_ref: Sequence[float] | None = None,
    penalty_weight: float = 1.0,
) -> dict[str, Any]:
    """Quality-Constrained POSU.

    Q-POSU = POSU - penalty_weight × QualityPenalty

    A molecule with high POSU but terrible quality gets penalized.
    This prevents "gaming" the site metrics by sacrificing drug-likeness.
    """
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    posu_scores = []
    for m in molecules:
        if m is not None:
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                posu_scores.append(compute_posu(m, site_map)["posu"])
            except Exception:
                posu_scores.append(0.0)
        else:
            posu_scores.append(0.0)
    mean_posu = float(np.mean(posu_scores)) if posu_scores else 0.0

    quality = compute_quality_penalty(
        molecules, reference_molecules,
        vina_scores=vina_scores, vina_ref=vina_ref,
    )
    q_posu = mean_posu - penalty_weight * quality["total_penalty"]

    return {
        "q_posu": q_posu,
        "mean_posu": mean_posu,
        "quality_penalty": quality["total_penalty"],
        "quality_components": quality["components"],
        "qed_mean": quality["qed_mean"],
        "mw_mean": quality["mw_mean"],
        "n_valid": quality["n_valid"],
        "n_total": quality["n_total"],
    }
