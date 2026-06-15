#!/usr/bin/env python3
"""Comprehensive evaluation + chart generation for KAG experiment suite.

Reads generated molecules from results/experiment_suite/ and produces:
  1. Full metrics table (traditional + new) with Wilcoxon p-values
  2. Box plots: min_dist_centroid per condition per pocket
  3. CDF plots: avg_COS per condition
  4. Pareto scatter plots: strain vs avg_COS (per pocket + combined)
  5. Summary table for paper

Usage:
    python scripts/evaluate_and_plot.py --results-dir results/experiment_suite --output-dir results/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from metrics_new import (
    compute_mol_centroid, compute_centroid_hew_distances,
    compute_cos, compute_e_site,
    classify_hew_environment, env_to_idx,
)

SITE_MAPS_DIR = ROOT / "experiments/targetdiff_replication/site_maps"

# For Wilcoxon test
try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_molecules(sdf_path):
    """Load molecules from SDF file."""
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    mols = []
    for m in supplier:
        if m is not None:
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                                 ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass
            mols.append(m)
    return mols


def compute_sa_score(mol):
    """Synthetic Accessibility score (simplified)."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer.calculateScore(mol)
    except ImportError:
        mw = Descriptors.MolWt(mol)
        n_rot = Descriptors.NumRotatableBonds(mol)
        return (mw / 200.0) + (n_rot / 5.0) + 1.0  # rough proxy


def evaluate_condition(sdf_path, site_map):
    """Evaluate all molecules in a condition."""
    mols = load_molecules(sdf_path)
    if not mols:
        return {"n": 0, "error": "No molecules loaded"}

    hew_sites = [s for s in site_map.get("sites", [])
                 if s.get("site_type") == "high_energy_water"]
    hew_centers = np.array([s["center"] for s in hew_sites])
    hew_env_indices = np.array([env_to_idx(classify_hew_environment(s)) for s in hew_sites])

    results = []
    for mol_idx, mol in enumerate(mols):
        centroid = compute_mol_centroid(mol)
        cd = compute_centroid_hew_distances(centroid, hew_centers) if centroid is not None else {}
        cos = compute_cos(mol, hew_centers, hew_env_indices, sigma=1.5)
        e_site = compute_e_site(mol, hew_centers, hew_env_indices, sigma=3.0, tau=10.0)

        try:
            qed_val = QED.qed(mol)
        except Exception:
            qed_val = float("nan")

        try:
            sa_val = compute_sa_score(mol)
        except Exception:
            sa_val = float("nan")

        results.append({
            "mol_id": mol_idx,
            "n_atoms": mol.GetNumAtoms(),
            **cd,
            **cos,
            "E_site": e_site,
            "QED": qed_val,
            "SA": sa_val,
        })

    if not results:
        return {"n": 0, "error": "No results"}

    # Aggregate statistics
    keys = ["min_dist_centroid", "avg_dist_centroid", "avg_COS", "max_COS", "E_site", "QED", "SA"]
    stats = {}
    for key in keys:
        vals = []
        for r in results:
            v = r.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
        if vals:
            stats[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "n": len(vals),
            }

    # Raw arrays for plotting
    raw = {}
    for key in keys:
        raw[key] = np.array([r.get(key, float("nan")) for r in results])

    return {"n": len(mols), "statistics": stats, "per_mol": results, "raw": raw}


def wilcoxon_pvalue(values_a, values_b):
    """Compute Mann-Whitney U p-value (Wilcoxon rank-sum)."""
    if not HAS_SCIPY:
        return float("nan")
    # Remove NaN
    a = np.array([v for v in values_a if not np.isnan(v)])
    b = np.array([v for v in values_b if not np.isnan(v)])
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def make_boxplot(data_dict, metric, output_path, title=None):
    """Generate a box plot comparing conditions for a metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(data_dict.keys())
    all_data = []
    labels = []
    for c in conditions:
        vals = data_dict[c].get("raw", {}).get(metric, np.array([]))
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            all_data.append(valid)
            labels.append(c)

    if not all_data:
        print(f"  No data for boxplot: {metric}")
        return

    fig, ax = plt.subplots(figsize=(len(all_data) * 1.5 + 2, 5))
    bp = ax.boxplot(all_data, labels=labels, patch_artist=True,
                    widths=0.5, showfliers=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for patch, color in zip(bp["boxes"], colors[:len(all_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title or f"{metric} by Condition", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Boxplot → {output_path}")


def make_cdf(data_dict, metric, output_path, title=None):
    """Generate CDF plot comparing conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (cond, data) in enumerate(data_dict.items()):
        vals = data.get("raw", {}).get(metric, np.array([]))
        valid = vals[~np.isnan(vals)]
        if len(valid) < 2:
            continue
        sorted_vals = np.sort(valid)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=cond, color=colors[i % len(colors)],
                linewidth=2, drawstyle="steps-post")

    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title(title or f"CDF of {metric}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ CDF → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results/experiment_suite"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/evaluation"))
    parser.add_argument("--pockets", type=str, default="3mfw,2gni,6o4x,2jke,2gqn,6phx")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pockets = [p.strip() for p in args.pockets.split(",")]
    conditions = ["unguided", "full_gradient", "hard_fix", "kag"]

    all_results = {}  # pocket → condition → evaluation dict

    for pocket in pockets:
        site_map_path = SITE_MAPS_DIR / f"{pocket}_site_map.json"
        if not site_map_path.exists():
            print(f"  ⚠ No site map for {pocket}")
            continue
        site_map = json.loads(site_map_path.read_text())

        pocket_dir = results_dir / pocket
        if not pocket_dir.exists():
            print(f"  ⚠ No results for {pocket}")
            continue

        pocket_results = {}
        for cond in conditions:
            sdf_path = pocket_dir / cond / "molecules.sdf"
            if not sdf_path.exists():
                continue
            print(f"  Evaluating {pocket}/{cond} ...")
            ev = evaluate_condition(sdf_path, site_map)
            ev["condition"] = cond
            ev["pocket"] = pocket
            pocket_results[cond] = ev

            stats = ev.get("statistics", {})
            n = ev.get("n", 0)
            print(f"    {n} mols, min_dist={stats.get('min_dist_centroid', {}).get('mean', 0):.2f}±{stats.get('min_dist_centroid', {}).get('std', 0):.2f}, "
                  f"COS={stats.get('avg_COS', {}).get('mean', 0):.4f}, E_site={stats.get('E_site', {}).get('mean', 0):.4f}")

        all_results[pocket] = pocket_results

    if not all_results:
        print("No results found!")
        return

    # ── Generate per-pocket charts ──
    for pocket, pdata in all_results.items():
        pocket_out = output_dir / pocket
        pocket_out.mkdir(parents=True, exist_ok=True)

        # Boxplot: min_dist_centroid
        if len(pdata) >= 2:
            make_boxplot(pdata, "min_dist_centroid",
                         pocket_out / "boxplot_min_dist_centroid.png",
                         f"{pocket}: Min Centroid Distance to HEW")

            make_boxplot(pdata, "avg_COS",
                         pocket_out / "boxplot_avg_COS.png",
                         f"{pocket}: Average COS")

            make_cdf(pdata, "avg_COS",
                     pocket_out / "cdf_avg_COS.png",
                     f"{pocket}: CDF of avg_COS")

    # ── Combined Pareto plot ──
    make_combined_pareto(all_results, output_dir / "pareto_strain_vs_cos_all.png")

    # ── Wilcoxon p-value table ──
    print(f"\n{'='*100}")
    print("Wilcoxon (Mann-Whitney U) p-values: KAG vs Unguided")
    print(f"{'='*100}")
    print(f"{'Pocket':<10} {'min_dist_cent':>14} {'avg_COS':>12} {'E_site':>12} {'QED':>12}")
    print("-" * 100)
    for pocket in pockets:
        if pocket not in all_results:
            continue
        pdata = all_results[pocket]
        if "kag" not in pdata or "unguided" not in pdata:
            continue
        kag_raw = pdata["kag"].get("raw", {})
        ug_raw = pdata["unguided"].get("raw", {})
        p_md = wilcoxon_pvalue(kag_raw.get("min_dist_centroid", []),
                                ug_raw.get("min_dist_centroid", []))
        p_cos = wilcoxon_pvalue(kag_raw.get("avg_COS", []),
                                 ug_raw.get("avg_COS", []))
        p_es = wilcoxon_pvalue(kag_raw.get("E_site", []),
                                ug_raw.get("E_site", []))
        p_qed = wilcoxon_pvalue(kag_raw.get("QED", []),
                                 ug_raw.get("QED", []))
        sig = lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"{pocket:<10} {p_md:.4f}{sig(p_md):<3} {p_cos:.4f}{sig(p_cos):<3} "
              f"{p_es:.4f}{sig(p_es):<3} {p_qed:.4f}{sig(p_qed):<3}")

    # ── Full summary table ──
    summary_table = {}
    for pocket in pockets:
        if pocket not in all_results:
            continue
        summary_table[pocket] = {}
        for cond in conditions:
            if cond not in all_results[pocket]:
                continue
            s = all_results[pocket][cond].get("statistics", {})
            summary_table[pocket][cond] = {
                k: (s[k]["mean"], s[k]["std"], s[k]["n"])
                for k in ["min_dist_centroid", "avg_COS", "E_site", "QED", "SA"]
                if k in s
            }

    # Save JSON
    summary_path = output_dir / "summary_table.json"
    with open(summary_path, "w") as f:
        json.dump(summary_table, f, indent=2, default=str)
    print(f"\n✓ Summary → {summary_path}")

    # Save CSV
    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, "w") as f:
        f.write("pocket,condition,min_dist_mean,min_dist_std,avg_COS_mean,avg_COS_std,"
                "E_site_mean,E_site_std,QED_mean,QED_std,SA_mean,SA_std,n\n")
        for pocket in summary_table:
            for cond in summary_table[pocket]:
                r = summary_table[pocket][cond]
                f.write(f"{pocket},{cond},"
                        f"{r.get('min_dist_centroid', ('','',''))[0]},{r.get('min_dist_centroid', ('','',''))[1]},"
                        f"{r.get('avg_COS', ('','',''))[0]},{r.get('avg_COS', ('','',''))[1]},"
                        f"{r.get('E_site', ('','',''))[0]},{r.get('E_site', ('','',''))[1]},"
                        f"{r.get('QED', ('','',''))[0]},{r.get('QED', ('','',''))[1]},"
                        f"{r.get('SA', ('','',''))[0]},{r.get('SA', ('','',''))[1]},"
                        f"{r.get('min_dist_centroid', ('','',0))[2]}\n")
    print(f"✓ CSV → {csv_path}")


def make_combined_pareto(all_results, output_path):
    """Generate combined Pareto scatter: strain vs avg_COS across all pockets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = {"unguided": "#1f77b4", "full_gradient": "#ff7f0e",
              "hard_fix": "#d62728", "kag": "#2ca02c"}
    markers = {"unguided": "o", "full_gradient": "s",
               "hard_fix": "^", "kag": "D"}

    for idx, (pocket, pdata) in enumerate(sorted(all_results.items())):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break

        for cond, ev in pdata.items():
            raw = ev.get("raw", {})
            strains = raw.get("strain", np.array([]))
            cos_vals = raw.get("avg_COS", np.array([]))
            if len(strains) == 0 or len(cos_vals) == 0:
                continue
            valid = ~(np.isnan(strains) | np.isnan(cos_vals))
            if valid.sum() < 2:
                continue
            ax.scatter(strains[valid], cos_vals[valid],
                       c=colors.get(cond, "#333"), marker=markers.get(cond, "o"),
                       alpha=0.6, s=25, label=cond)

        ax.set_xlabel("Strain", fontsize=10)
        ax.set_ylabel("avg_COS", fontsize=10)
        ax.set_title(pocket, fontsize=12)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pareto: Strain vs avg_COS (all pockets)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Combined Pareto → {output_path}")


if __name__ == "__main__":
    main()
