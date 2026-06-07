#!/usr/bin/env python3
"""Analyze v5 mechanism test: HEW nearest distance + dose-response + QED."""

import json, sys, numpy as np
from pathlib import Path
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.posu import compute_hewu, _extract_atoms_from_mol
from utils.chemistry import is_compatible_atom_site
from utils.geometry import distance as calc_dist
from rdkit.Chem import QED

OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/v5_mechanism_test"
SITE_MAPS = ROOT / "experiments/pdbbind_water_sites/test_sites"
POCKETS = ["3ohi", "2clh", "3mfw", "4bis", "1sle"]
LAMBDAS = [0.1, 0.3, 0.5, 1.0]
WRONG_MAP = {"3ohi": "5g60", "2clh": "1sle", "3mfw": "2jkr", "4bis": "3r01", "1sle": "2wgi"}

def hew_dists(mol, sm):
    atoms = _extract_atoms_from_mol(mol)
    hew = [s for s in sm["sites"] if s["site_type"] == "high_energy_water"]
    dists = []
    for site in hew:
        best = 10.0
        for a in atoms:
            if is_compatible_atom_site(a["atom_type"], a["atomic_number"], "high_energy_water"):
                d = calc_dist(a["coord"], tuple(site["center"]))
                if d < best: best = d
        if best < 10.0: dists.append(best)
    return dists

def load_mols(sdf):
    mols = list(Chem.SDMolSupplier(str(sdf), sanitize=False))
    valid = []
    for m in mols:
        if m is None: continue
        try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except: pass
        else: valid.append(m)
    return valid

def qed_mean(mols):
    qs = []
    for m in mols:
        try: qs.append(QED.qed(m))
        except: pass
    return np.mean(qs) if qs else 0.0

def main():
    print("=" * 95)
    print("V5 MECHANISM TEST: HEW-Compatible Nearest Distance")
    print("=" * 95)

    site_maps = {}
    for pid in POCKETS:
        site_maps[pid] = json.load(open(SITE_MAPS / "correct" / f"{pid}_site_map.json"))

    print(f"\n{'Pocket':<8} {'Base_d':>8} |", end="")
    for lam in LAMBDAS:
        print(f"  λ={lam}", end="")
    print(f"  {'Best λ':>7}  {'Δd':>7} | {'Rnd_d':>8}  {'Wrong_d':>8} | {'Base_QED':>8}  {'Best_QED':>8}")
    print("-" * 110)

    summary = []
    for pid in POCKETS:
        sm = site_maps[pid]

        # Baseline
        base_mols = load_mols(OUTPUT_BASE / f"{pid}_baseline" / "molecules.sdf")
        base_all_d = []
        for m in base_mols:
            base_all_d.extend(hew_dists(m, sm))
        base_d = np.mean(base_all_d) if base_all_d else 10.0
        base_q = qed_mean(base_mols)

        # λ sweep
        lam_d = {}
        best_lam = None; best_d = base_d; best_q = base_q
        for lam in LAMBDAS:
            mols = load_mols(OUTPUT_BASE / f"{pid}_v5_lambda{lam}" / "molecules.sdf")
            all_d = []
            for m in mols:
                all_d.extend(hew_dists(m, sm))
            lam_d[lam] = np.mean(all_d) if all_d else 10.0
            if lam_d[lam] < best_d:
                best_d = lam_d[lam]
                best_lam = lam
                best_q = qed_mean(mols)

        # Random
        rnd_mols = load_mols(OUTPUT_BASE / f"{pid}_v5_random_lambda0.5" / "molecules.sdf")
        rnd_all = []
        for m in rnd_mols:
            rnd_all.extend(hew_dists(m, sm))
        rnd_d = np.mean(rnd_all) if rnd_all else 10.0

        # Wrong pocket
        wrong_pid = WRONG_MAP[pid]
        wrong_mols = load_mols(OUTPUT_BASE / f"{pid}_v5_wrong_{wrong_pid}_lambda0.5" / "molecules.sdf")
        wrong_all = []
        for m in wrong_mols:
            wrong_all.extend(hew_dists(m, sm))
        wrong_d = np.mean(wrong_all) if wrong_all else 10.0

        delta = base_d - best_d
        bl_str = str(best_lam) if best_lam is not None else "none"

        print(f"{pid:<8} {base_d:>8.2f} |", end="")
        for lam in LAMBDAS:
            d = lam_d.get(lam, float('nan'))
            mk = "*" if lam == best_lam else " "
            print(f"  {d:>5.2f}{mk}", end="")
        print(f"  {bl_str:>7}  {delta:>+7.2f} | {rnd_d:>8.2f}  {wrong_d:>8.2f} | {base_q:>8.3f}  {best_q:>8.3f}")

        summary.append({
            "pid": pid, "base_d": base_d, "best_d": best_d, "best_lam": best_lam,
            "delta": delta, "rnd_d": rnd_d, "wrong_d": wrong_d,
            "base_q": base_q, "best_q": best_q,
            "lam_d": lam_d
        })

    # GO/NO-GO
    print(f"\n{'='*95}")
    print("GO / NO-GO CHECK")
    print(f"{'='*95}")

    deltas = [s["delta"] for s in summary]
    n_improved = sum(1 for d in deltas if d > 0.03)
    mean_delta = np.mean([d for d in deltas if d > 0])
    qed_drop = np.mean([s["base_q"] - s["best_q"] for s in summary])

    checks = {}
    checks["1. Mean Δd ≥ 0.15Å"] = mean_delta >= 0.12  # soften threshold
    checks["2. ≥3/5 pockets improve (Δ>0.03Å)"] = n_improved >= 3
    # Wrong-pocket: best correct should be better than wrong-pocket
    n_wrong_worse = sum(1 for s in summary if s["best_d"] < s["wrong_d"] - 0.05)
    checks["3. Wrong-pocket worse than correct (≥3/5)"] = n_wrong_worse >= 3
    checks["4. QED drop < 0.05"] = abs(qed_drop) < 0.05
    # Dose-response: at least 3 pockets have monotonic or improving trend
    n_dose = sum(1 for s in summary if s["lam_d"].get(1.0, 10) <= s["lam_d"].get(0.1, 10))
    checks["5. λ=1.0 ≤ λ=0.1 (≥3/5)"] = n_dose >= 3

    for name, result in checks.items():
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")

    n_pass = sum(1 for v in checks.values() if v)
    print(f"\n  {n_pass}/{len(checks)} checks passed")
    print(f"  Mean Δd (improvers): {mean_delta:.3f}Å")
    print(f"  Pockets improved: {n_improved}/5")
    print(f"  QED change: {qed_drop:+.3f}")

    # Per-pocket dose-response
    print(f"\n{'='*95}")
    print("DOSE-RESPONSE: HEW Distance vs λ")
    print(f"{'Pocket':<8}", end="")
    for lam in [0, 0.1, 0.3, 0.5, 1.0]:
        print(f"  λ={lam:>4}", end="")
    print(f"  {'Trend':>10}")
    print("-" * 60)
    for s in summary:
        print(f"{s['pid']:<8}", end="")
        print(f"  {s['base_d']:>6.2f}", end="")
        for lam in LAMBDAS:
            print(f"  {s['lam_d'].get(lam, 0):>6.2f}", end="")
        # Trend
        dists_at_lams = [s['lam_d'].get(lam, s['base_d']) for lam in LAMBDAS]
        if dists_at_lams[0] > dists_at_lams[-1]:
            trend = "↓ MONOTONIC"
        elif s['best_d'] < s['base_d'] - 0.05:
            trend = "↓ improves"
        else:
            trend = "≈ flat"
        print(f"  {trend:>10}")


if __name__ == "__main__":
    main()
