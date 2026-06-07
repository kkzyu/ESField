#!/usr/bin/env python3
"""Analyze ESField experiment matrix results.

Computes all paper metrics from generated SDFs:
  Step 1: SBR, SQD, POSU distributions
  Step 2: RSS (Random-Site Sensitivity)
  Step 3: Paired t-tests, per-site-type breakdown
  Step 4: Pareto frontier, Q-POSU vs λ

Usage:
  python scripts/analyze_experiments.py              # All steps
  python scripts/analyze_experiments.py --step 1     # Step 1 only
  python scripts/analyze_experiments.py --step 4 --plot  # Step 4 with plots
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.posu import compute_posu, compute_hewu, compute_swdp, compute_hcfu
from evaluation.site_blindness import compute_sbr, compute_sqd, compute_rss
from evaluation.quality_constrained import compute_q_posu, compute_quality_penalty

OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/experiment_matrix"
TEST_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"
SITE_MAPS_DIR = ROOT / "experiments/pdbbind_water_sites/test_sites"

ABLATION_POCKETS = ["5g60", "3b28", "6ayn", "3r01", "1sle"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_molecules(sdf_path):
    mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False))
    valid = []
    for m in mols:
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            valid.append(m)
        except Exception:
            pass
    return valid


def load_site_map(pocket_id, condition="correct"):
    path = SITE_MAPS_DIR / condition / f"{pocket_id}_site_map.json"
    return json.loads(path.read_text())


def get_pocket_ids():
    return [p["pdb_id"] for p in json.loads(TEST_POCKETS_JSON.read_text())]


def paired_ttest(values_a, values_b):
    """Paired t-test. Returns (t_stat, p_value, cohens_d)."""
    a = np.array(values_a)
    b = np.array(values_b)
    diffs = a - b
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    if std_diff == 0:
        return float("nan"), float("nan"), 0.0
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    # Two-sided p from t-distribution
    from math import gamma
    def _tcdf(t, df):
        """Approximate t CDF using regularized incomplete beta."""
        x = df / (df + t * t)
        from math import exp, log
        # Use scipy if available, otherwise approximate
        try:
            from scipy.stats import t as tdist
            return tdist.cdf(t, df)
        except ImportError:
            # Approximation for p-value
            pass
        return float("nan")
    try:
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(a, b)
    except ImportError:
        p_value = float("nan")
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    return float(t_stat), float(p_value), float(cohens_d)


# ---------------------------------------------------------------------------
# Step 1: Blindness Diagnosis
# ---------------------------------------------------------------------------

def analyze_step1():
    """Compute SBR, SQD, POSU for baseline molecules across 20 test pockets."""
    print("\n" + "=" * 70)
    print("Step 1: Opportunity-Blindness Diagnosis")
    print("=" * 70)

    pocket_ids = get_pocket_ids()
    results = []

    print(f"{'Pocket':<8} {'#Mol':>6} {'POSU':>8} {'SBR':>8} {'SQD':>8} {'r(Vina)':>9}")
    print("-" * 60)

    for pid in pocket_ids:
        sdf_path = OUTPUT_BASE / "step1" / f"{pid}_baseline" / "molecules.sdf"
        if not sdf_path.exists():
            print(f"{pid:<8} {'SKIP':>6}")
            continue

        site_map = load_site_map(pid, "correct")
        mols = load_molecules(sdf_path)
        if len(mols) < 5:
            print(f"{pid:<8} {len(mols):>6} (too few)")
            continue

        # Compute POSU for all molecules
        posu_vals = [compute_posu(m, site_map)["posu"] for m in mols]
        mean_posu = np.mean(posu_vals)

        # SBR (without Vina since we don't have it yet — use QED only)
        sbr_result = compute_sbr(mols, site_map, qed_threshold=0.4)

        # SQD placeholder (needs Vina scores)
        sqd_str = "N/A"
        sqd_result = {"sqd": float("nan")}

        results.append({
            "pocket": pid,
            "n_mols": len(mols),
            "mean_posu": mean_posu,
            "posu_std": float(np.std(posu_vals)),
            "sbr": sbr_result["sbr"],
            "n_good": sbr_result["n_good"],
            "n_blind": sbr_result["n_blind"],
        })

        print(f"{pid:<8} {len(mols):>6} {mean_posu:>8.4f} {sbr_result['sbr']:>8.1%} "
              f"{sqd_str:>8}")

    if len(results) < 2:
        print("\nInsufficient data for summary.")
        return results

    sbr_vals = [r["sbr"] for r in results if not np.isnan(r["sbr"])]
    posu_vals = [r["mean_posu"] for r in results]
    print(f"\nSummary ({len(results)} pockets):")
    print(f"  Mean SBR:  {np.mean(sbr_vals):.1%} ± {np.std(sbr_vals):.1%}" if sbr_vals else "  SBR: N/A")
    print(f"  Mean POSU: {np.mean(posu_vals):.4f} ± {np.std(posu_vals):.4f}")
    print(f"  Overall good molecules: {sum(r['n_good'] for r in results)}")
    print(f"  Overall blind molecules: {sum(r['n_blind'] for r in results)}")

    return results


# ---------------------------------------------------------------------------
# Step 2: Site Physical Meaningfulness
# ---------------------------------------------------------------------------

def analyze_step2():
    """Compute RSS across 3 site map conditions for 5 ablation pockets."""
    print("\n" + "=" * 70)
    print("Step 2: Site Physical Meaningfulness (RSS)")
    print("=" * 70)

    conditions = ["correct", "random", "shuffled"]
    results = []

    print(f"{'Pocket':<8} {'POSU_cor':>10} {'POSU_rnd':>10} {'POSU_shf':>10} {'RSS':>8} {'Δrand':>8} {'Δshuf':>8}")
    print("-" * 70)

    for pid in ABLATION_POCKETS:
        site_map = load_site_map(pid, "correct")
        cond_posu = {}
        all_data = True

        for cond in conditions:
            sdf_path = OUTPUT_BASE / "step2" / f"{pid}_{cond}" / "molecules.sdf"
            if not sdf_path.exists():
                all_data = False
                break
            mols = load_molecules(sdf_path)
            posu_vals = [compute_posu(m, site_map)["posu"] for m in mols]
            cond_posu[cond] = (mols, posu_vals)

        if not all_data:
            print(f"{pid:<8} SKIP (missing data)")
            continue

        posu_correct = np.mean(cond_posu["correct"][1])
        posu_random = np.mean(cond_posu["random"][1])
        posu_shuffled = np.mean(cond_posu["shuffled"][1])

        delta_rand = posu_correct - posu_random
        delta_shuf = posu_correct - posu_shuffled
        rss = min(delta_rand, delta_shuf)

        results.append({
            "pocket": pid,
            "posu_correct": float(posu_correct),
            "posu_random": float(posu_random),
            "posu_shuffled": float(posu_shuffled),
            "rss": float(rss),
            "delta_rand": float(delta_rand),
            "delta_shuf": float(delta_shuf),
        })

        print(f"{pid:<8} {posu_correct:>10.4f} {posu_random:>10.4f} "
              f"{posu_shuffled:>10.4f} {rss:>8.4f} {delta_rand:>8.4f} {delta_shuf:>8.4f}")

    if results:
        rss_vals = [r["rss"] for r in results]
        posu_c_vals = [r["posu_correct"] for r in results]
        posu_r_vals = [r["posu_random"] for r in results]
        posu_s_vals = [r["posu_shuffled"] for r in results]

        print(f"\nSummary ({len(results)} pockets):")
        print(f"  Mean POSU correct:  {np.mean(posu_c_vals):.4f} ± {np.std(posu_c_vals):.4f}")
        print(f"  Mean POSU random:   {np.mean(posu_r_vals):.4f} ± {np.std(posu_r_vals):.4f}")
        print(f"  Mean POSU shuffled: {np.mean(posu_s_vals):.4f} ± {np.std(posu_s_vals):.4f}")
        print(f"  Mean RSS: {np.mean(rss_vals):.4f} ± {np.std(rss_vals):.4f}")

        # Paired t-test: correct vs random
        t, p, d = paired_ttest(posu_c_vals, posu_r_vals)
        print(f"  Correct vs Random: t={t:.3f}, p={p:.4f}, d={d:.3f}" if not np.isnan(t) else
              f"  Correct vs Random: N/A (need scipy)")

    return results


# ---------------------------------------------------------------------------
# Step 3: ESField Improvement
# ---------------------------------------------------------------------------

def analyze_step3():
    """Paired comparison of baseline vs λ=0.5 vs λ=1.0 vs random.

    Main claim: POSU_guided > POSU_baseline (statistically significant).
    """
    print("\n" + "=" * 70)
    print("Step 3: ESField Improvement")
    print("=" * 70)

    pocket_ids = get_pocket_ids()
    conditions = ["baseline", "lambda0.5", "lambda1.0", "random"]
    all_results = []

    print(f"\n{'Pocket':<8} {'POSU_base':>10} {'POSU_0.5':>10} {'POSU_1.0':>10} "
          f"{'POSU_rnd':>10} {'Δ1.0':>8} {'HEWU_base':>10} {'HEWU_1.0':>10}")
    print("-" * 85)

    for pid in pocket_ids:
        site_map = load_site_map(pid, "correct")
        cond_data = {}
        all_ok = True

        for cond in conditions:
            sdf_path = OUTPUT_BASE / "step3" / f"{pid}_{cond}" / "molecules.sdf"
            if not sdf_path.exists():
                all_ok = False
                break
            mols = load_molecules(sdf_path)
            posu_vals = [compute_posu(m, site_map) for m in mols]
            hewu_vals = [compute_hewu(m, site_map) for m in mols]
            cond_data[cond] = {
                "mols": mols,
                "posu": [p["posu"] for p in posu_vals],
                "hewu": [h["mean_utility"] for h in hewu_vals],
            }

        if not all_ok:
            print(f"{pid:<8} SKIP (missing data)")
            continue

        posu_base = np.mean(cond_data["baseline"]["posu"])
        posu_05 = np.mean(cond_data["lambda0.5"]["posu"])
        posu_10 = np.mean(cond_data["lambda1.0"]["posu"])
        posu_rnd = np.mean(cond_data["random"]["posu"])
        hewu_base = np.mean(cond_data["baseline"]["hewu"])
        hewu_10 = np.mean(cond_data["lambda1.0"]["hewu"])

        delta = posu_10 - posu_base

        all_results.append({
            "pocket": pid,
            "posu_baseline": float(posu_base),
            "posu_lambda0.5": float(posu_05),
            "posu_lambda1.0": float(posu_10),
            "posu_random": float(posu_rnd),
            "delta_posu": float(delta),
            "hewu_baseline": float(hewu_base),
            "hewu_lambda1.0": float(hewu_10),
        })

        print(f"{pid:<8} {posu_base:>10.4f} {posu_05:>10.4f} {posu_10:>10.4f} "
              f"{posu_rnd:>10.4f} {delta:>+8.4f} {hewu_base:>10.4f} {hewu_10:>10.4f}")

    if len(all_results) < 5:
        print("\nInsufficient data for statistical tests (need ≥5 pockets).")
        return all_results

    # Paired t-tests
    posu_base = [r["posu_baseline"] for r in all_results]
    posu_05 = [r["posu_lambda0.5"] for r in all_results]
    posu_10 = [r["posu_lambda1.0"] for r in all_results]
    posu_rnd = [r["posu_random"] for r in all_results]

    print(f"\n{'='*70}")
    print("Statistical Tests (paired t-test, two-sided)")
    print(f"{'='*70}")

    tests = [
        ("Baseline vs λ=0.5", posu_base, posu_05),
        ("Baseline vs λ=1.0", posu_base, posu_10),
        ("Baseline vs Random", posu_base, posu_rnd),
        ("λ=1.0 vs Random", posu_10, posu_rnd),
    ]

    for name, a, b in tests:
        t, p, d = paired_ttest(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {name:<25} t={t:>7.3f}  p={p:.4f} {sig}  d={d:.3f}")

    # Mean improvement
    deltas = [r["delta_posu"] for r in all_results]
    n_improved = sum(1 for d in deltas if d > 0)
    print(f"\n  Pockets with POSU improvement: {n_improved}/{len(deltas)} "
          f"({n_improved/len(deltas):.0%})")
    print(f"  Mean ΔPOSU (λ=1.0 - baseline): {np.mean(deltas):.4f} ± {np.std(deltas, ddof=1)/np.sqrt(len(deltas)):.4f} (SEM)")

    # HEWRU improvement
    hewu_base = [r["hewu_baseline"] for r in all_results]
    hewu_10 = [r["hewu_lambda1.0"] for r in all_results]
    t_h, p_h, d_h = paired_ttest(hewu_10, hewu_base)
    sig_h = "***" if p_h < 0.001 else "**" if p_h < 0.01 else "*" if p_h < 0.05 else "n.s."
    print(f"  HEWRU: λ=1.0 vs baseline  t={t_h:.3f}  p={p_h:.4f} {sig_h}  d={d_h:.3f}")

    return all_results


# ---------------------------------------------------------------------------
# Step 4: Quality-Constrained Validation
# ---------------------------------------------------------------------------

def analyze_step4():
    """λ sweep: find Pareto frontier, compute Q-POSU."""
    print("\n" + "=" * 70)
    print("Step 4: Quality-Constrained Validation (λ sweep)")
    print("=" * 70)

    LAMBDA_VALS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    step4_pockets = get_pocket_ids()[:10]

    # Aggregate per-λ across pockets
    lambda_summary = {lam: {"posu": [], "qed": [], "mw": [], "valid": [], "collapse": []}
                      for lam in LAMBDA_VALS}

    print(f"\n{'λ':>6} {'POSU':>8} {'QED':>8} {'MW':>8} {'Q-POSU':>10} {'Valid%':>8} {'n_pockets':>10}")
    print("-" * 65)

    for lam in LAMBDA_VALS:
        for pid in step4_pockets:
            sdf_path = OUTPUT_BASE / "step4" / f"{pid}_lambda{lam}" / "molecules.sdf"
            if not sdf_path.exists():
                continue

            site_map = load_site_map(pid, "correct")
            mols = load_molecules(sdf_path)
            if len(mols) < 3:
                continue

            # POSU
            posu_vals = [compute_posu(m, site_map)["posu"] for m in mols]
            lambda_summary[lam]["posu"].append(np.mean(posu_vals))

            # QED
            from rdkit.Chem import QED, Descriptors
            qeds, mws = [], []
            for m in mols:
                try:
                    qeds.append(QED.qed(m))
                    mws.append(Descriptors.MolWt(m))
                except Exception:
                    pass
            if qeds:
                lambda_summary[lam]["qed"].append(np.mean(qeds))
            if mws:
                lambda_summary[lam]["mw"].append(np.mean(mws))

            # Validity from metadata
            meta_path = OUTPUT_BASE / "step4" / f"{pid}_lambda{lam}" / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                valid_pct = meta.get("metrics", {}).get("valid", 0) / max(1, meta.get("metrics", {}).get("total", 20))
                lambda_summary[lam]["valid"].append(valid_pct)

    for lam in LAMBDA_VALS:
        s = lambda_summary[lam]
        n = len(s["posu"])
        if n == 0:
            continue
        mean_posu = np.mean(s["posu"])
        mean_qed = np.mean(s["qed"]) if s["qed"] else 0.0
        mean_mw = np.mean(s["mw"]) if s["mw"] else 0.0
        mean_valid = np.mean(s["valid"]) if s["valid"] else 0.0

        # Q-POSU (simplified: POSU - QED_drop)
        qed_drop = max(0.0, 0.5 - mean_qed) / 0.5  # reference QED=0.5
        q_posu = mean_posu - qed_drop

        print(f"{lam:>6.1f} {mean_posu:>8.4f} {mean_qed:>8.4f} {mean_mw:>8.0f} "
              f"{q_posu:>10.4f} {mean_valid:>7.1%} {n:>10}")

    # Find sweet spot (highest Q-POSU)
    best_lam = None
    best_qposu = -float("inf")
    for lam in LAMBDA_VALS:
        s = lambda_summary[lam]
        if not s["posu"]:
            continue
        mean_posu = np.mean(s["posu"])
        mean_qed = np.mean(s["qed"]) if s["qed"] else 0.0
        qed_drop = max(0.0, 0.5 - mean_qed) / 0.5
        q_posu = mean_posu - qed_drop
        if q_posu > best_qposu:
            best_qposu = q_posu
            best_lam = lam

    print(f"\nSweet spot: λ={best_lam} (Q-POSU={best_qposu:.4f})")

    # Check for mode collapse at high λ
    print(f"\nQuality at λ ≥ 1.5:")
    for lam in [1.5, 2.0]:
        s = lambda_summary[lam]
        if s["qed"]:
            print(f"  λ={lam:.1f}: QED={np.mean(s['qed']):.3f} ± {np.std(s['qed']):.3f}, "
                  f"Valid={np.mean(s['valid']):.1%}")

    return lambda_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze ESField experiment results")
    parser.add_argument("--step", default="all",
                        choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (requires matplotlib)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    all_data = {}

    if args.step in ("1", "all"):
        all_data["step1"] = analyze_step1()
    if args.step in ("2", "all"):
        all_data["step2"] = analyze_step2()
    if args.step in ("3", "all"):
        all_data["step3"] = analyze_step3()
    if args.step in ("4", "all"):
        all_data["step4"] = analyze_step4()

    if args.output:
        # Convert numpy types for JSON
        import json as _json
        class NpEncoder(_json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        with open(args.output, "w") as f:
            _json.dump(all_data, f, cls=NpEncoder, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
