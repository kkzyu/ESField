#!/usr/bin/env python3
"""Analyze Vina score trends: occupied vs non-occupied molecules (Phase II).

For each pocket:
  1. Load Vina scores and occupancy labels
  2. Split into occupied vs non-occupied groups
  3. Mann-Whitney U test + Cliff's delta effect size
  4. Box plot + scatter plot
  5. Output statistical summary

Usage:
    PYTHONPATH=src python scripts/analyze_energy_trend.py \
        --pocket-id 2gni \
        --docking-csv phase2_results/2gni_vina_scores.csv \
        --sdf-file experiments/v71_full_study/2gni_v7.1.sdf \
        --site-map experiments/.../2gni_site_map.json \
        --output-dir phase2_results/
"""

import argparse, csv, json, os, sys
from pathlib import Path

import numpy as np
from scipy import stats

ESFIELD_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ESFIELD_ROOT, "src"))

from evaluation.site_occupancy import direct_occupancy_rate
from guidance.latent_guidance import classify_hew_environment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def compute_occupancy_per_molecule(sdf_path, site_map, threshold=2.5):
    """Compute per-molecule occupancy boolean from SDF.

    Returns list of bool (True = occupies ≥1 HEW site with compatible atom).
    """
    from rdkit import Chem
    from evaluation.site_occupancy import _get_atom_info, _HEW_COMPATIBLE_ELEMENTS

    mols = list(Chem.SDMolSupplier(sdf_path, sanitize=False))

    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return [False] * len(mols)

    per_mol_occ = []
    for mol in mols:
        if mol is None:
            per_mol_occ.append(False)
            continue
        try:
            atoms = _get_atom_info(mol)
        except Exception:
            per_mol_occ.append(False)
            continue

        occupied = False
        for site in hew_sites:
            env = classify_hew_environment(site)
            center = tuple(site["center"])
            compat_elements = _HEW_COMPATIBLE_ELEMENTS.get(env, set())
            for atom in atoms:
                d = np.sqrt(sum((a - b) ** 2 for a, b in zip(atom["coord"], center)))
                if d <= threshold and atom["atomic_num"] in compat_elements:
                    occupied = True
                    break
            if occupied:
                break
        per_mol_occ.append(occupied)

    return per_mol_occ


def cliff_delta(x, y):
    """Cliff's delta effect size for two independent samples.

    δ = P(X > Y) - P(X < Y), ranges [-1, 1].
    |δ| < 0.147: negligible, < 0.33: small, < 0.474: medium, ≥ 0.474: large.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    greater = sum(1 for xi in x for yi in y if xi > yi)
    lesser = sum(1 for xi in x for yi in y if xi < yi)
    return (greater - lesser) / (nx * ny)


def cohens_d(x, y):
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    if sp < 1e-8:
        return 0.0
    return (np.mean(x) - np.mean(y)) / sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-id", required=True)
    parser.add_argument("--docking-csv", required=True)
    parser.add_argument("--sdf-file", required=True,
                        help="SDF with molecular coordinates (e.g., minimized)")
    parser.add_argument("--occupancy-sdf", default=None,
                        help="Original SDF for occupancy labeling (if different from --sdf-file)")
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-group-size", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    site_map = json.load(open(args.site_map))

    # Load Vina scores
    vina_scores = {}
    with open(args.docking_csv) as f:
        for row in csv.DictReader(f):
            idx = int(row["mol_index"])
            if row["success"] == "True" and row["vina_score"]:
                vina_scores[idx] = float(row["vina_score"])

    print(f"Loaded {len(vina_scores)} Vina scores for {args.pocket_id}")

    # Compute per-molecule occupancy (use original SDF if provided, else the input SDF)
    occ_sdf = args.occupancy_sdf or args.sdf_file
    occ_per_mol = compute_occupancy_per_molecule(occ_sdf, site_map)
    if args.occupancy_sdf:
        print(f"Occupancy labels from: {args.occupancy_sdf}")

    # Group: occupied vs non-occupied
    occ_scores = []
    nonocc_scores = []
    for idx in range(len(occ_per_mol)):
        if idx not in vina_scores:
            continue
        if occ_per_mol[idx]:
            occ_scores.append(vina_scores[idx])
        else:
            nonocc_scores.append(vina_scores[idx])

    print(f"Occupied: {len(occ_scores)} molecules, mean Vina = {np.mean(occ_scores):.1f} kcal/mol")
    print(f"Non-occupied: {len(nonocc_scores)} molecules, mean Vina = {np.mean(nonocc_scores):.1f} kcal/mol")

    result = {
        "pocket": args.pocket_id,
        "n_occ": len(occ_scores),
        "n_nonocc": len(nonocc_scores),
        "mean_score_occ": float(np.mean(occ_scores)) if occ_scores else None,
        "mean_score_nonocc": float(np.mean(nonocc_scores)) if nonocc_scores else None,
        "median_score_occ": float(np.median(occ_scores)) if occ_scores else None,
        "median_score_nonocc": float(np.median(nonocc_scores)) if nonocc_scores else None,
        "std_score_occ": float(np.std(occ_scores)) if occ_scores else None,
        "std_score_nonocc": float(np.std(nonocc_scores)) if nonocc_scores else None,
    }

    # Statistical test (only if both groups have enough samples)
    if len(occ_scores) >= args.min_group_size and len(nonocc_scores) >= args.min_group_size:
        # Mann-Whitney U (two-sided)
        u_stat, p_value = stats.mannwhitneyu(occ_scores, nonocc_scores, alternative="two-sided")
        delta = cliff_delta(occ_scores, nonocc_scores)
        d_val = cohens_d(occ_scores, nonocc_scores)

        result["mann_whitney_u"] = float(u_stat)
        result["p_value"] = float(p_value)
        result["cliff_delta"] = float(delta)
        result["cohens_d"] = float(d_val)

        pct_better = sum(1 for s in occ_scores if s < np.median(nonocc_scores)) / len(occ_scores)
        result["pct_occ_better_than_nonocc_median"] = float(pct_better)

        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "n.s."))
        result["significance"] = sig

        print(f"Mann-Whitney U={u_stat:.1f}, p={p_value:.4f} {sig}")
        print(f"Cliff's delta={delta:.3f}, Cohen's d={d_val:.3f}")
        print(f"% occupied better than non-occ median: {pct_better*100:.1f}%")

        # ── Box plot ──
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot([occ_scores, nonocc_scores], labels=["Occupied", "Non-occupied"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#FF9800")

        # Significance annotation
        if p_value < 0.05:
            y_max = max(max(occ_scores), max(nonocc_scores))
            ax.annotate(sig, xy=(1.5, y_max * 1.02), ha="center", fontsize=14, fontweight="bold")

        ax.set_ylabel("Vina Score (kcal/mol, lower is better)")
        ax.set_title(f"{args.pocket_id}: Occupied vs Non-occupied\n"
                     f"(n={len(occ_scores)} vs {len(nonocc_scores)}, p={p_value:.3f})")
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"{args.pocket_id}_boxplot.png"), dpi=150)
        fig.savefig(os.path.join(args.output_dir, f"{args.pocket_id}_boxplot.pdf"))
        plt.close()

        # ── Scatter plot (best compat distance vs Vina score) ──
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#4CAF50" if occ_per_mol[i] else "#FF9800" for i in vina_scores.keys()]
        scores_list = list(vina_scores.values())
        ax.scatter(range(len(scores_list)), scores_list, c=colors, alpha=0.6, edgecolors="black", linewidth=0.3)
        ax.axhline(y=np.median(occ_scores) if occ_scores else 0, color="#4CAF50", linestyle="--", label=f"Occ median ({np.median(occ_scores):.1f})")
        ax.axhline(y=np.median(nonocc_scores) if nonocc_scores else 0, color="#FF9800", linestyle="--", label=f"Non-occ median ({np.median(nonocc_scores):.1f})")
        ax.set_xlabel("Molecule index")
        ax.set_ylabel("Vina Score (kcal/mol)")
        ax.set_title(f"{args.pocket_id}: Vina Scores (green=occupied, orange=non-occupied)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"{args.pocket_id}_scatter.png"), dpi=150)
        plt.close()
    else:
        result["p_value"] = None
        result["significance"] = "insufficient_data"
        print(f"Insufficient data: need ≥{args.min_group_size} per group, got {len(occ_scores)} vs {len(nonocc_scores)}")

    # Save result
    result_path = os.path.join(args.output_dir, f"{args.pocket_id}_energy_trend.json")
    json.dump(result, open(result_path, "w"), indent=2)
    print(f"Saved to {result_path}")
    return result


if __name__ == "__main__":
    main()
