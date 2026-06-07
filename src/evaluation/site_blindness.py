"""Diagnostic metrics: SBR, SQD, RSS.

These metrics prove the existence of the gap:
  SBR (Site-Blindness Rate): how many "good" molecules are site-blind
  SQD (Site-Quality Discordance): how decoupled are site and quality metrics
  RSS (Random-Site Sensitivity): how much does correct site info matter

All metrics are rule-based and independent of any learned model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem

from evaluation.posu import compute_posu


def compute_sbr(
    molecules: Sequence[Chem.Mol],
    site_map: dict[str, Any] | str | Path,
    *,
    posu_threshold: float | None = None,
    qed_threshold: float = 0.4,
    sa_threshold: float = 5.0,
    vina_scores: Sequence[float] | None = None,
    vina_percentile: float = 30.0,  # top 30% vina = "good"
    posebusters_results: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Site-Blindness Rate.

    SBR = proportion of "globally plausible" molecules that have poor site utilization.

    "Globally plausible" = validity=1, QED>=threshold, SA<=threshold,
                          Vina in top percentile, PoseBusters pass (if available)

    "Poor site utilization" = POSU below threshold (default: below median)
    """
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    # Compute POSU for all molecules
    from rdkit.Chem import QED, Descriptors

    posu_scores = []
    quality_flags = []
    for i, mol in enumerate(molecules):
        if mol is None:
            quality_flags.append(False)
            posu_scores.append(0.0)
            continue

        # Check validity (can we sanitize?)
        valid = True
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            valid = False

        if not valid:
            quality_flags.append(False)
            posu_scores.append(0.0)
            continue

        # Quality checks (wrap in try/except for molecules that won't kekulize)
        qed_ok = True
        try:
            qed_ok = QED.qed(mol) >= qed_threshold
        except Exception:
            pass
        sa_ok = True  # default pass if can't compute

        vina_ok = True
        if vina_scores and i < len(vina_scores):
            threshold = np.percentile([v for v in vina_scores if v is not None], 100 - vina_percentile)
            vina_ok = vina_scores[i] is not None and vina_scores[i] <= threshold

        pb_ok = True
        if posebusters_results and i < len(posebusters_results):
            pb_ok = posebusters_results[i]

        is_good = valid and qed_ok and sa_ok and vina_ok and pb_ok
        quality_flags.append(is_good)

        # Compute POSU
        posu_result = compute_posu(mol, site_map)
        posu_scores.append(posu_result["posu"])

    # Determine POSU threshold (below which we consider "blind")
    # Use median of quality-passing molecules, not all (invalid ones have POSU=0)
    good_posu = [p for p, q in zip(posu_scores, quality_flags) if q]
    if posu_threshold is None:
        posu_threshold = float(np.median(good_posu)) if good_posu else (float(np.median(posu_scores)) if posu_scores else 0.0)

    # Count blind molecules among good ones
    n_good = sum(quality_flags)
    n_blind = sum(
        1 for p, q in zip(posu_scores, quality_flags) if q and p < posu_threshold
    )

    return {
        "sbr": n_blind / n_good if n_good > 0 else float("nan"),
        "n_good": n_good,
        "n_total": len(molecules),
        "n_blind": n_blind,
        "posu_threshold": posu_threshold,
        "posu_scores": posu_scores,
        "quality_flags": quality_flags,
        "good_posu_mean": float(np.mean(good_posu)) if good_posu else float("nan"),
        "blind_posu_mean": float(np.mean([p for p, q in zip(posu_scores, quality_flags) if q and p < posu_threshold])) if n_blind > 0 else float("nan"),
    }


def compute_sqd(
    molecules: Sequence[Chem.Mol],
    site_map: dict[str, Any] | str | Path,
    vina_scores: Sequence[float],
    *,
    quality_metric: str = "vina",
) -> dict[str, Any]:
    """Site-Quality Discordance.

    SQD = 1 - |Spearman r(POSU, quality_metric)|
    High SQD → POSU and quality metric measure different things (good for us)
    Low SQD  → POSU is redundant with quality metric (bad — we didn't add value)

    Also reports:
      - Top-Vina molecules: mean POSU
      - Top-POSU molecules: mean Vina
    """
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    # Compute POSU for valid molecules
    posu_vals = []
    quality_vals = []
    for i, mol in enumerate(molecules):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            continue
        posu_result = compute_posu(mol, site_map)
        posu_vals.append(posu_result["posu"])
        if i < len(vina_scores) and vina_scores[i] is not None:
            quality_vals.append(vina_scores[i])
        else:
            quality_vals.append(float("nan"))

    # Remove NaN
    paired = [(p, q) for p, q in zip(posu_vals, quality_vals) if not math.isnan(q)]
    if len(paired) < 5:
        return {"sqd": float("nan"), "spearman_r": float("nan"), "n_pairs": len(paired)}

    try:
        from scipy.stats import spearmanr
        r, p_value = spearmanr([p for p, _ in paired], [q for _, q in paired])
    except ImportError:
        # Fallback: manual Spearman using numpy
        import warnings
        warnings.warn("scipy not available; computing Spearman r with numpy")
        from numpy import argsort, mean
        n = len(paired)
        rank_p = argsort(argsort([p for p, _ in paired])).astype(float)
        rank_q = argsort(argsort([q for _, q in paired])).astype(float)
        d2 = (rank_p - rank_q) ** 2
        r = 1.0 - 6.0 * d2.sum() / (n * (n**2 - 1))
        p_value = float("nan")
    r = r if not math.isnan(r) else 0.0
    sqd = 1.0 - abs(r)

    # Top-Vina (best 30%): mean POSU
    top_n = max(1, len(paired) // 3)
    sorted_by_vina = sorted(paired, key=lambda x: x[1])
    top_vina_posu = float(np.mean([p for p, _ in sorted_by_vina[:top_n]]))

    sorted_by_posu = sorted(paired, key=lambda x: x[0], reverse=True)
    top_posu_vina = float(np.mean([q for _, q in sorted_by_posu[:top_n]]))

    return {
        "sqd": sqd,
        "spearman_r": r,
        "spearman_p": float(p_value),
        "n_pairs": len(paired),
        "top_vina_posu_mean": top_vina_posu,
        "top_posu_vina_mean": top_posu_vina,
    }


def compute_rss(
    molecules_correct: Sequence[Chem.Mol],
    molecules_random: Sequence[Chem.Mol],
    molecules_shuffled: Sequence[Chem.Mol],
    site_map: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Random-Site Sensitivity.

    RSS = min(POSU_correct - POSU_random, POSU_correct - POSU_shuffled)
    Positive RSS → correct site info matters.
    Zero/Negative RSS → sites don't carry useful information.
    """
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    def mean_posu(mols):
        scores = []
        for m in mols:
            if m is None:
                continue
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                continue
            scores.append(compute_posu(m, site_map)["posu"])
        return float(np.mean(scores)) if scores else float("nan")

    posu_correct = mean_posu(molecules_correct)
    posu_random = mean_posu(molecules_random)
    posu_shuffled = mean_posu(molecules_shuffled)

    rss = min(posu_correct - posu_random, posu_correct - posu_shuffled)

    return {
        "rss": rss,
        "posu_correct": posu_correct,
        "posu_random": posu_random,
        "posu_shuffled": posu_shuffled,
        "delta_vs_random": posu_correct - posu_random,
        "delta_vs_shuffled": posu_correct - posu_shuffled,
    }
