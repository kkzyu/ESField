#!/usr/bin/env python3
"""Analyze sum_norm guidance 5-pocket experiment.

Primary metric: HEW-compatible nearest atom |d - 3.0|.
Go/No-Go criteria (≥3/5 needed):
  1. 3mfw or 1sle at least one improves from flat
  2. sum_norm mean |d-3.0| better than old sum
  3. sum_norm top-10% |d-3.0| better than old sum
  4. wrong-pocket doesn't produce equal improvement
  5. QED drop < 0.03

Usage:
  python scripts/analyze_sum_norm_experiment.py
"""

import json, sys, numpy as np
warnings = __import__('warnings')
warnings.filterwarnings("ignore")
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.posu import _extract_atoms_from_mol
from utils.chemistry import is_compatible_atom_site
from utils.geometry import distance as calc_dist

OUT = ROOT / "experiments/pdbbind_water_sites/v5_sum_norm_test"
SM = ROOT / "experiments/pdbbind_water_sites/test_sites"
D0 = 3.0
POCKETS = ["3ohi", "2clh", "3mfw", "4bis", "1sle"]

CONDITIONS = [
    ("baseline", 0.0, "sum", "correct"),
    ("v5-sum_lambda1.0", 1.0, "sum", "correct"),
    ("v5-sumnorm_lambda1.0", 1.0, "sum_norm", "correct"),
    ("v5-sumnorm_lambda5.0", 5.0, "sum_norm", "correct"),
    ("v5-sumnorm_lambda10.0", 10.0, "sum_norm", "correct"),
    ("v5-sumnorm_lambda20.0", 20.0, "sum_norm", "correct"),
    ("v5-sumnorm_wrong_lambda10.0", 10.0, "sum_norm", "wrong"),
]


def hew_deviation(mol, sm_dict):
    """Best |d - 3.0| across all HEW-compatible atom-site pairs."""
    atoms = _extract_atoms_from_mol(mol)
    hew_sites = [s for s in sm_dict["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return 10.0
    best = 10.0
    for site in hew_sites:
        sc = site["center"]
        for a in atoms:
            if is_compatible_atom_site(a["atom_type"], a["atomic_number"], "high_energy_water"):
                d = calc_dist(a["coord"], tuple(sc))
                dev = abs(d - D0)
                if dev < best:
                    best = dev
    return best


def hew_raw_distance(mol, sm_dict):
    """Shortest HEW-compatible atom distance (no D0 offset)."""
    atoms = _extract_atoms_from_mol(mol)
    hew_sites = [s for s in sm_dict["sites"] if s["site_type"] == "high_energy_water"]
    best = 10.0
    for site in hew_sites:
        sc = site["center"]
        for a in atoms:
            if is_compatible_atom_site(a["atom_type"], a["atomic_number"], "high_energy_water"):
                d = calc_dist(a["coord"], tuple(sc))
                if d < best:
                    best = d
    return best


def load_mols(pid, cond_name):
    sdf_path = OUT / f"{pid}_{cond_name}" / "molecules.sdf"
    if not sdf_path.exists():
        return []
    mols = [m for m in Chem.SDMolSupplier(str(sdf_path), sanitize=False) if m is not None]
    for m in mols:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            pass
    return mols


def load_qed(pid, cond_name):
    meta_path = OUT / f"{pid}_{cond_name}" / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta["metrics"]["qed_mean"]
    return None


def main():
    # Load all data
    print("Loading molecules...")
    data = {}  # {(pid, cond_name): [deviations, raw_distances, qed]}
    for pid in POCKETS:
        sm_dict = json.load(open(SM / "correct" / f"{pid}_site_map.json"))
        for cond_name, lam, agg, site_type in CONDITIONS:
            key = f"{pid}_{cond_name}"
            mols = load_mols(pid, cond_name)
            if not mols:
                print(f"  WARNING: {key} — no molecules found")
                continue
            devs = [hew_deviation(m, sm_dict) for m in mols]
            raws = [hew_raw_distance(m, sm_dict) for m in mols]
            qed = load_qed(pid, cond_name)
            data[(pid, cond_name)] = (devs, raws, qed, len(mols))

    # ================================================
    # TABLE 1: Main results — |d-3.0| per condition
    # ================================================
    print(f"\n{'='*120}")
    print("TABLE 1: HEW-Compatible |d-3.0| (mean ± std)")
    print(f"{'='*120}")
    header = f"  {'Pocket':<8} {'Baseline':>12} {'Sum λ=1':>12} {'SN λ=1':>12} {'SN λ=5':>12} {'SN λ=10':>12} {'SN λ=20':>12} {'Wrong SN10':>12}"
    print(header)
    print("  " + "-" * 118)

    all_means = {pid: {} for pid in POCKETS}
    for pid in POCKETS:
        row = f"  {pid:<8}"
        for cond_name, lam, agg, site_type in CONDITIONS:
            key = (pid, cond_name)
            if key in data:
                devs, raws, qed, n = data[key]
                mean_d = np.mean(devs)
                all_means[pid][cond_name] = mean_d
                row += f" {mean_d:>11.3f}"
            else:
                row += f" {'N/A':>11}"
        print(row)

    # ================================================
    # TABLE 2: Δ from baseline
    # ================================================
    print(f"\n{'='*120}")
    print("TABLE 2: Δ |d-3.0| from baseline (negative = better)")
    print(f"{'='*120}")
    print(header)
    print("  " + "-" * 118)
    for pid in POCKETS:
        row = f"  {pid:<8}"
        base_key = (pid, "baseline")
        if base_key not in data:
            row += " N/A" * 7
            print(row)
            continue
        base_dev = np.mean(data[base_key][0])
        row += f" {0.0:>11.3f}"  # baseline delta is 0
        for cond_name, lam, agg, site_type in CONDITIONS[1:]:  # skip baseline
            key = (pid, cond_name)
            if key in data:
                mean_d = np.mean(data[key][0])
                delta = mean_d - base_dev
                row += f" {delta:>+11.3f}"
            else:
                row += f" {'N/A':>11}"
        print(row)

    # ================================================
    # TABLE 3: Top-10% |d-3.0| (best molecules)
    # ================================================
    print(f"\n{'='*120}")
    print("TABLE 3: Top-10% |d-3.0| (best 2 of 20 molecules, mean)")
    print(f"{'='*120}")
    print(header)
    print("  " + "-" * 118)
    for pid in POCKETS:
        row = f"  {pid:<8}"
        for cond_name, lam, agg, site_type in CONDITIONS:
            key = (pid, cond_name)
            if key in data:
                devs = data[key][0]
                top2 = sorted(devs)[:2]
                row += f" {np.mean(top2):>11.3f}"
            else:
                row += f" {'N/A':>11}"
        print(row)

    # ================================================
    # TABLE 4: QED
    # ================================================
    print(f"\n{'='*120}")
    print("TABLE 4: QED Mean")
    print(f"{'='*120}")
    print(header)
    print("  " + "-" * 118)
    qed_deltas = []
    for pid in POCKETS:
        row = f"  {pid:<8}"
        base_qed = data.get((pid, "baseline"), (None, None, None, 0))[2]
        for cond_name, lam, agg, site_type in CONDITIONS:
            key = (pid, cond_name)
            if key in data:
                qed = data[key][2]
                row += f" {qed:>11.3f}" if qed is not None else f" {'N/A':>11}"
                if base_qed is not None and qed is not None and cond_name != "baseline":
                    qed_deltas.append(qed - base_qed)
            else:
                row += f" {'N/A':>11}"
        print(row)

    # ================================================
    # GO/NO-GO CRITERIA
    # ================================================
    print(f"\n{'='*120}")
    print("GO/NO-GO ASSESSMENT (≥3/5 needed)")
    print(f"{'='*120}")

    go_count = 0

    # Criterion 1: 3mfw or 1sle improve from flat
    print("\n--- Criterion 1: 3mfw or 1sle improve ---")
    improved = False
    for pid in ["3mfw", "1sle"]:
        base_key = (pid, "baseline")
        if base_key not in data:
            continue
        base = np.mean(data[base_key][0])
        for cond_name, lam, agg, site_type in CONDITIONS:
            if agg == "sum_norm" and site_type == "correct":
                key = (pid, cond_name)
                if key in data:
                    delta = np.mean(data[key][0]) - base
                    print(f"  {pid} {cond_name}: Δ = {delta:+.3f}")
                    if delta < -0.02:
                        improved = True
    if improved:
        print("  => PASS: At least one previously-flat pocket improved")
        go_count += 1
    else:
        print("  => FAIL: No improvement in 3mfw or 1sle")

    # Criterion 2: sum_norm mean |d-3.0| better than old sum
    print("\n--- Criterion 2: sum_norm mean better than old sum ---")
    sum_better = 0
    for pid in POCKETS:
        sum_key = (pid, "v5-sum_lambda1.0")
        sn_keys = [(pid, c[0]) for c in CONDITIONS if c[2] == "sum_norm" and c[3] == "correct"]
        if sum_key not in data:
            continue
        sum_mean = np.mean(data[sum_key][0])
        best_sn = min((np.mean(data[k][0]), k) for k in sn_keys if k in data)
        delta = best_sn[0] - sum_mean
        status = "✓" if delta < 0 else "✗"
        print(f"  {pid}: sum={sum_mean:.3f}, best sum_norm={best_sn[0]:.3f} ({best_sn[1][1]}) → Δ={delta:+.3f} {status}")
        if delta < 0:
            sum_better += 1
    if sum_better >= 3:
        print(f"  => PASS: {sum_better}/5 pockets sum_norm better")
        go_count += 1
    else:
        print(f"  => FAIL: only {sum_better}/5")

    # Criterion 3: sum_norm top-10% better than old sum top-10%
    print("\n--- Criterion 3: sum_norm top-10% better than old sum ---")
    top_better = 0
    for pid in POCKETS:
        sum_key = (pid, "v5-sum_lambda1.0")
        sn_keys = [(pid, c[0]) for c in CONDITIONS if c[2] == "sum_norm" and c[3] == "correct"]
        if sum_key not in data:
            continue
        sum_top2 = np.mean(sorted(data[sum_key][0])[:2])
        best_sn_top2 = min((np.mean(sorted(data[k][0])[:2]), k) for k in sn_keys if k in data)
        delta = best_sn_top2[0] - sum_top2
        status = "✓" if delta < 0 else "✗"
        print(f"  {pid}: sum top2={sum_top2:.3f}, best sum_norm top2={best_sn_top2[0]:.3f} → Δ={delta:+.3f} {status}")
        if delta < 0:
            top_better += 1
    if top_better >= 3:
        print(f"  => PASS: {top_better}/5 pockets top-10% better")
        go_count += 1
    else:
        print(f"  => FAIL: only {top_better}/5")

    # Criterion 4: wrong-pocket doesn't produce equal improvement
    print("\n--- Criterion 4: wrong-pocket worse than correct ---")
    wrong_check = 0
    for pid in POCKETS:
        correct_key = (pid, "v5-sumnorm_lambda10.0")
        wrong_key = (pid, "v5-sumnorm_wrong_lambda10.0")
        if correct_key in data and wrong_key in data:
            corr = np.mean(data[correct_key][0])
            wrg = np.mean(data[wrong_key][0])
            status = "✓" if corr <= wrg else "✗"
            print(f"  {pid}: correct={corr:.3f}, wrong={wrg:.3f} {status}")
            if corr <= wrg:
                wrong_check += 1
    if wrong_check >= 3:
        print(f"  => PASS: {wrong_check}/5 correct ≤ wrong")
        go_count += 1
    else:
        print(f"  => FAIL: only {wrong_check}/5")

    # Criterion 5: QED drop < 0.03
    print("\n--- Criterion 5: QED stability ---")
    if qed_deltas:
        max_drop = min(qed_deltas)
        mean_delta = np.mean(qed_deltas)
        print(f"  Max QED drop: {max_drop:.3f}, Mean QED change: {mean_delta:.3f}")
        if max_drop > -0.03:
            print(f"  => PASS: QED drop {max_drop:.3f} < 0.03")
            go_count += 1
        else:
            print(f"  => FAIL: QED drop {max_drop:.3f} ≥ 0.03")
    else:
        print("  No QED data available")

    # ================================================
    # FINAL VERDICT
    # ================================================
    print(f"\n{'='*120}")
    print(f"GO COUNT: {go_count}/5")
    if go_count >= 3:
        print("VERDICT: **GO** — proceed with sum_norm guidance")
    else:
        print("VERDICT: **NO-GO** — consider ESField as reranking method")
    print(f"{'='*120}")

    # ================================================
    # TABLE 5: Raw HEW distance (for reference)
    # ================================================
    print(f"\n{'='*120}")
    print("TABLE 5 (Reference): HEW-Compatible Nearest Distance (Å, mean)")
    print(f"{'='*120}")
    print(header)
    print("  " + "-" * 118)
    for pid in POCKETS:
        row = f"  {pid:<8}"
        for cond_name, lam, agg, site_type in CONDITIONS:
            key = (pid, cond_name)
            if key in data:
                raws = data[key][1]
                row += f" {np.mean(raws):>11.2f}"
            else:
                row += f" {'N/A':>11}"
        print(row)


if __name__ == "__main__":
    main()
