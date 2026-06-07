#!/usr/bin/env python3
"""v6-D.2 Actionable-Pocket Validation Test.

4 pockets × 6 conditions, paired seeds, per-site analysis.

Usage:
  python scripts/run_v6d2_actionable_test.py --dry-run
  python scripts/run_v6d2_actionable_test.py
  python scripts/run_v6d2_actionable_test.py --analyze-only
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drugflow_esfield_guide import (
    load_esfield_potential, load_drugflow_model, generate_molecules,
    POTENTIAL_DEFAULT_CKPT, DRUGFLOW_CKPT,
)
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from evaluation.posu import compute_posu, classify_hew_environment, is_compatible_hew_v2
from utils.chemistry import infer_atom_type, atomic_number
from utils.geometry import distance as calc_distance
from models.analytic_esfield import HEW_ENV_HYDROPHOBIC, HEW_ENV_BURIED

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POCKETS = {
    "2gni":  {"class": "B_ACTIONABLE", "wrong": "5g60"},
    "6o4x":  {"class": "B_ACTIONABLE", "wrong": "1sle"},
    "3mfw":  {"class": "B_ACTIONABLE_HARD", "wrong": "2jkr"},
    "2clh":  {"class": "A_CEILING_CONTROL", "wrong": "1sle"},
}

CONDITIONS = [
    ("baseline",           "learned_v5", 0.0, "correct"),
    ("v6d2_lambda0.5",     "analytic_v6d2", 0.5, "correct"),
    ("v6d2_lambda1.0",     "analytic_v6d2", 1.0, "correct"),
    ("v6d2_lambda2.0",     "analytic_v6d2", 2.0, "correct"),
    ("v6d2_wrong_lambda1.0","analytic_v6d2", 1.0, "wrong"),
    ("v6d2_random_lambda1.0","analytic_v6d2", 1.0, "random"),
]

V6D2_CONFIG = {
    "sigma_cap": 2.5, "sigma_occ": 1.0,
    "min_confidence": 0.7, "top_k": 5,
    "enabled_envs": ("hydrophobic",),
    "wrong_atom_weight": 0.5, "overfill_weight": 0.3,
    "softmax_tau": 1.0,
    "guidance_start": 0.3, "guidance_end": 0.88,
}

GEN_PARAMS = {"num_samples": 20, "timesteps": 40, "gen_batch_size": 5, "device": "cuda:0"}

TEST_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"
SITE_MAPS_DIR = ROOT / "experiments/pdbbind_water_sites/test_sites"
OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/v6d2_actionable_test"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pocket_paths(pdb_id):
    test_pockets = json.loads(TEST_POCKETS_JSON.read_text())
    for p in test_pockets:
        if p["pdb_id"] == pdb_id:
            pd = Path(p["dir"])
            return {
                "protein_pdb": str(pd / f"{pdb_id}_protein.pdb"),
                "pocket_pdb": str(pd / f"{pdb_id}_pocket.pdb"),
                "ref_ligand": str(pd / f"{pdb_id}_ligand.sdf"),
            }
    raise ValueError(f"Pocket {pdb_id} not found")

def load_progress():
    path = OUTPUT_BASE / "progress.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}

def save_progress(prog):
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BASE / "progress.json", "w") as f:
        json.dump(prog, f, indent=2)

def load_mols(sdf_path):
    if not Path(sdf_path).exists():
        return []
    mols = []
    for m in Chem.SDMolSupplier(sdf_path, sanitize=False):
        if m is None: continue
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            mols.append(m)
        except: pass
    return mols

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(model, potential, progress, dry_run=False):
    all_runs = []
    for pid, pinfo in POCKETS.items():
        for cond_name, guide_type, lam, site_src in CONDITIONS:
            all_runs.append((pid, cond_name, guide_type, lam, site_src))

    if dry_run:
        print(f"Would run {len(all_runs)} conditions ({len(all_runs)*20} generations)")
        for pid, cn, gt, lam, ss in all_runs[:12]:
            print(f"  {pid}: {cn} (λ={lam}, guide={gt}, site={ss})")
        return

    n_done = sum(1 for pid, cn, _, _, _ in all_runs if progress.get(f"{pid}_{cn}") == "done")
    n_total = len(all_runs)
    if n_done == n_total:
        print("All complete.")
        return

    print(f"{n_total - n_done} remaining of {n_total}")

    for pid, cond_name, guide_type, lam, site_src in all_runs:
        key = f"{pid}_{cond_name}"
        if progress.get(key) == "done":
            continue
        sdf_path = OUTPUT_BASE / key / "molecules.sdf"
        if sdf_path.exists() and list(Chem.SDMolSupplier(str(sdf_path), sanitize=False)):
            progress[key] = "done"
            save_progress(progress)
            continue

        paths = get_pocket_paths(pid)

        if site_src == "wrong":
            wrong_pid = POCKETS[pid]["wrong"]
            site_map = str(SITE_MAPS_DIR / "correct" / f"{wrong_pid}_site_map.json")
        elif site_src == "random":
            site_map = str(SITE_MAPS_DIR / "random" / f"{pid}_site_map.json")
        else:
            site_map = str(SITE_MAPS_DIR / "correct" / f"{pid}_site_map.json")

        seed = hash(f"{pid}_{cond_name}") % 100000
        print(f"\n[{n_done+1}/{n_total}] {pid} {cond_name} (λ={lam}, seed={seed})")
        t0 = time.time()
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=site_map,
                output_dir=str(OUTPUT_BASE / key),
                esfield_lambda=lam, guide_type=guide_type,
                v6_config=V6D2_CONFIG if guide_type == "analytic_v6d2" else None,
                seed=seed, **GEN_PARAMS,
            )
            print(f"  Valid: {result['valid_count']}/20, QED: {result['metrics']['qed_mean']:.3f}, "
                  f"{time.time()-t0:.0f}s")
            progress[key] = "done"
            save_progress(progress)
            n_done += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nDone: {n_done}/{n_total}")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def get_candidate_hews(sm):
    """Extract high-confidence hydrophobic candidate HEW sites."""
    cand = []
    for s in sm["sites"]:
        if s["site_type"] != "high_energy_water": continue
        env = classify_hew_environment(s)
        conf = s.get("confidence", 1.0)
        if env == HEW_ENV_HYDROPHOBIC and conf >= 0.7:
            cand.append(s)
    return cand

def compute_mol_site_metrics(mol, hew_sites):
    """Per-molecule, per-site metrics."""
    conf_obj = mol.GetConformer()
    results = []
    for hs in hew_sites:
        sc = tuple(hs["center"])
        best_cap = 0.0; best_occ = 0.0
        min_cd = float("inf"); min_cd_noncompat = float("inf")
        direct = 0
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 1: continue
            pos = conf_obj.GetAtomPosition(a.GetIdx())
            d = calc_distance((pos.x, pos.y, pos.z), sc)
            cap = np.exp(-d**2/(2*2.5**2))
            occ = np.exp(-d**2/(2*1.0**2))
            best_cap = max(best_cap, cap)
            best_occ = max(best_occ, occ)
            at = infer_atom_type(a.GetSymbol())
            an = atomic_number(a.GetSymbol())
            if is_compatible_hew_v2(at, an, hs):
                if d < min_cd:
                    min_cd = d
                if d < 2.0:
                    direct = 1
            else:
                if d < min_cd_noncompat:
                    min_cd_noncompat = d
        results.append({
            "best_cap": best_cap, "best_occ": best_occ,
            "min_compat_d": min_cd if min_cd < float("inf") else None,
            "min_noncompat_d": min_cd_noncompat if min_cd_noncompat < float("inf") else None,
            "direct_occ": direct,
        })
    return results

def analyze():
    print("\n" + "="*100)
    print("v6-D.2 ACTIONABLE-POCKET VALIDATION — ANALYSIS")
    print("="*100)

    all_results = {}

    for pid, pinfo in POCKETS.items():
        correct_sm = json.loads(open(SITE_MAPS_DIR / "correct" / f"{pid}_site_map.json").read())
        hews = get_candidate_hews(correct_sm)

        print(f"\n{'='*80}")
        print(f"Pocket: {pid} ({pinfo['class']}) — {len(hews)} candidate HEW")
        print(f"{'='*80}")

        # Per-condition stats
        cond_stats = {}
        for cond_name, guide_type, lam, site_src in CONDITIONS:
            key = f"{pid}_{cond_name}"
            sdf_path = OUTPUT_BASE / key / "molecules.sdf"
            mols = load_mols(sdf_path)

            # Use correct site map for analysis (even for wrong/random conditions)
            site_sm = correct_sm

            if not mols:
                cond_stats[cond_name] = {"n_mols": 0}
                continue

            qeds = []; posus = []; hewus = []
            all_site_metrics = []

            for mol in mols:
                try: qeds.append(QED.qed(mol))
                except: qeds.append(0.0)
                try:
                    r = compute_posu(mol, site_sm)
                    posus.append(r["posu"]); hewus.append(r["hew_mean"])
                except:
                    posus.append(0.0); hewus.append(0.0)
                all_site_metrics.append(compute_mol_site_metrics(mol, hews))

            # Aggregate site-level metrics
            n_sites = len(hews)
            site_stats = []
            for j in range(n_sites):
                caps = [sm[j]["best_cap"] for sm in all_site_metrics]
                occs = [sm[j]["best_occ"] for sm in all_site_metrics]
                cds = [sm[j]["min_compat_d"] for sm in all_site_metrics if sm[j]["min_compat_d"] is not None]
                directs = [sm[j]["direct_occ"] for sm in all_site_metrics]
                site_stats.append({
                    "site_id": j,
                    "conf": hews[j].get("confidence", 1.0),
                    "baseline_d": None,  # filled below for baseline
                    "mean_cap": np.mean(caps), "mean_occ": np.mean(occs),
                    "mean_compat_d": np.mean(cds) if cds else float("nan"),
                    "min_compat_d": np.min(cds) if cds else float("nan"),
                    "direct_occ_count": sum(directs),
                    "direct_occ_rate": sum(directs) / len(directs),
                })

            cond_stats[cond_name] = {
                "n_mols": len(mols),
                "qed_mean": np.mean(qeds), "qed_std": np.std(qeds),
                "posu_mean": np.mean(posus), "posu_std": np.std(posus),
                "hewu_mean": np.mean(hewus), "hewu_std": np.std(hewus),
                "site_stats": site_stats,
            }

        # Print per-condition summary
        print(f"\n{'Condition':<28s} {'#Mol':>5s} {'POSU':>7s} {'HEWU':>7s} {'QED':>7s} "
              f"{'CapScr':>7s} {'SoftOcc':>7s} {'MinCD(A)':>9s} {'DirectOcc':>10s}")
        print("-"*95)

        for cond_name, _, _, _ in CONDITIONS:
            s = cond_stats.get(cond_name)
            if s is None or s["n_mols"] == 0:
                print(f"{cond_name:<28s} NO DATA")
                continue
            caps = [st["mean_cap"] for st in s["site_stats"]]
            occs = [st["mean_occ"] for st in s["site_stats"]]
            mcds = [st["mean_compat_d"] for st in s["site_stats"] if not np.isnan(st["mean_compat_d"])]
            do_rate = sum(st["direct_occ_count"] for st in s["site_stats"]) / max(
                sum(len([sm for sm in []]) for _ in s["site_stats"]), 1)
            # Simplified direct occ rate
            do_total = sum(st["direct_occ_count"] for st in s["site_stats"])
            n_total_mol_site = len(hews) * s["n_mols"]
            do_rate = do_total / n_total_mol_site if n_total_mol_site > 0 else 0

            print(f"{cond_name:<28s} {s['n_mols']:>5d} {s['posu_mean']:>7.4f} {s['hewu_mean']:>7.4f} "
                  f"{s['qed_mean']:>7.3f} {np.mean(caps):>7.3f} {np.mean(occs):>7.3f} "
                  f"{np.mean(mcds):>9.2f} {do_total:>5d}/{n_total_mol_site:<5d}")

        # Per-site detail
        print(f"\n--- Per-Site Detail ---")
        base_ss = cond_stats.get("baseline", {}).get("site_stats", [])
        for cond_name, _, _, _ in CONDITIONS:
            s = cond_stats.get(cond_name)
            if s is None: continue
            for j, st in enumerate(s.get("site_stats", [])):
                base_d = base_ss[j]["mean_compat_d"] if j < len(base_ss) and not np.isnan(base_ss[j].get("mean_compat_d", float("nan"))) else float("nan")
                delta = ""
                if not np.isnan(st["mean_compat_d"]) and not np.isnan(base_d):
                    delta = f"Δ={st['mean_compat_d']-base_d:+.2f}Å"
                if cond_name == "baseline":
                    print(f"  Site{j} (conf={st['conf']:.2f}): d={st['mean_compat_d']:.2f}Å, "
                          f"Cap={st['mean_cap']:.3f}, Occ={st['mean_occ']:.3f}, "
                          f"DirRate={st['direct_occ_rate']:.1%}")
                else:
                    improved = ""
                    if not np.isnan(st["mean_compat_d"]) and not np.isnan(base_d) and st["mean_compat_d"] < base_d - 0.1:
                        improved = " ← IMPROVED"
                    print(f"  [{cond_name}] Site{j}: d={st['mean_compat_d']:.2f}Å {delta}, "
                          f"Cap={st['mean_cap']:.3f}, Occ={st['mean_occ']:.3f}, "
                          f"DirRate={st['direct_occ_rate']:.1%}{improved}")

        all_results[pid] = cond_stats

    # --- Cross-pocket summary ---
    print("\n" + "="*100)
    print("CROSS-POCKET SUMMARY")
    print("="*100)

    for metric_label, metric_key in [
        ("DirectOcc Rate", "direct_occ"),
        ("SoftOcc", "soft_occ"),
        ("MinCompatD", "min_compat_d"),
        ("HEWU", "hewu_mean"),
        ("QED", "qed_mean"),
    ]:
        print(f"\n--- {metric_label} ---")
        print(f"{'Pocket':<8s} {'class':<20s}", end="")
        for cn, _, _, _ in CONDITIONS:
            print(f" {cn:>20s}", end="")
        print()
        for pid, pinfo in POCKETS.items():
            s = all_results.get(pid, {})
            print(f"{pid:<8s} {pinfo['class']:<20s}", end="")
            for cn, _, _, _ in CONDITIONS:
                cs = s.get(cn, {})
                if not cs or cs["n_mols"] == 0:
                    print(f" {'N/A':>20s}", end="")
                elif metric_key == "direct_occ":
                    do = sum(st["direct_occ_count"] for st in cs.get("site_stats", []))
                    n_total = len(cs.get("site_stats", [])) * cs["n_mols"]
                    print(f" {do}/{n_total} ({do/n_total*100:5.1f}%)".rjust(21), end="")
                elif metric_key == "soft_occ":
                    occs = [st["mean_occ"] for st in cs.get("site_stats", [])]
                    print(f" {np.mean(occs):>20.3f}" if occs else f" {'N/A':>20s}", end="")
                elif metric_key == "min_compat_d":
                    mcds = [st["mean_compat_d"] for st in cs.get("site_stats", []) if not np.isnan(st.get("mean_compat_d", float("nan")))]
                    print(f" {np.mean(mcds):>20.2f}" if mcds else f" {'N/A':>20s}", end="")
                elif metric_key in ("hewu_mean", "qed_mean"):
                    print(f" {cs.get(metric_key, 0):>20.4f}", end="")
            print()

    # --- Go/No-Go Assessment ---
    print("\n" + "="*100)
    print("GO/NO-GO ASSESSMENT")
    print("="*100)

    for pid in POCKETS:
        s = all_results.get(pid, {})
        base = s.get("baseline", {})
        v6d2_l10 = s.get("v6d2_lambda1.0", {})
        rand = s.get("v6d2_random_lambda1.0", {})
        wrong = s.get("v6d2_wrong_lambda1.0", {})

        if not base or not v6d2_l10:
            print(f"\n{pid}: INCOMPLETE DATA")
            continue

        # DirectOcc rate
        n_hew = len(base.get("site_stats", []))
        base_do = sum(st["direct_occ_count"] for st in base.get("site_stats", []))
        base_n = n_hew * base["n_mols"]
        v6_do = sum(st["direct_occ_count"] for st in v6d2_l10.get("site_stats", []))
        v6_n = n_hew * v6d2_l10["n_mols"]
        rand_do = sum(st["direct_occ_count"] for st in rand.get("site_stats", [])) if rand else 0
        wrong_do = sum(st["direct_occ_count"] for st in wrong.get("site_stats", [])) if wrong else 0

        base_dor = base_do / base_n if base_n > 0 else 0
        v6_dor = v6_do / v6_n if v6_n > 0 else 0
        rand_dor = rand_do / v6_n if rand and v6_n > 0 else 0
        wrong_dor = wrong_do / v6_n if wrong and v6_n > 0 else 0

        # SoftOcc
        base_occ = np.mean([st["mean_occ"] for st in base.get("site_stats", [])])
        v6_occ = np.mean([st["mean_occ"] for st in v6d2_l10.get("site_stats", [])])
        rand_occ = np.mean([st["mean_occ"] for st in rand.get("site_stats", [])]) if rand else 0
        wrong_occ = np.mean([st["mean_occ"] for st in wrong.get("site_stats", [])]) if wrong else 0

        # MinCompatD
        base_d = np.mean([st["mean_compat_d"] for st in base.get("site_stats", []) if not np.isnan(st.get("mean_compat_d", float("nan")))])
        v6_d = np.mean([st["mean_compat_d"] for st in v6d2_l10.get("site_stats", []) if not np.isnan(st.get("mean_compat_d", float("nan")))])
        rand_d = np.mean([st["mean_compat_d"] for st in rand.get("site_stats", []) if rand and not np.isnan(st.get("mean_compat_d", float("nan")))]) if rand else float("nan")

        # QED
        base_qed = base.get("qed_mean", 0)
        v6_qed = v6d2_l10.get("qed_mean", 0)

        print(f"\n{pid} ({POCKETS[pid]['class']}):")
        print(f"  DirectOcc rate: base={base_dor:.1%} → v6d2={v6_dor:.1%} "
              f"(rand={rand_dor:.1%}, wrong={wrong_dor:.1%})")
        print(f"  SoftOcc:       base={base_occ:.3f} → v6d2={v6_occ:.3f} "
              f"(rand={rand_occ:.3f}, wrong={wrong_occ:.3f})")
        print(f"  MinCompatD:    base={base_d:.2f}Å → v6d2={v6_d:.2f}Å "
              f"(rand={rand_d:.2f}Å)" if not np.isnan(rand_d) else "")
        print(f"  QED:           base={base_qed:.3f} → v6d2={v6_qed:.3f}")

        # Go checks
        dor_ok = v6_dor >= base_dor + 0.10 or v6_dor >= 2 * base_dor
        occ_ok = v6_occ >= base_occ * 1.10
        d_ok = v6_d <= base_d - 0.3
        rand_ok = v6_dor > rand_dor and v6_occ > rand_occ
        wrong_ok = v6_dor > wrong_dor
        qed_ok = v6_qed >= base_qed - 0.03

        checks = [
            ("DirRate +10pp or 2×", dor_ok),
            ("SoftOcc +10%", occ_ok),
            ("MinCompatD -0.3Å", d_ok),
            ("v6d2 > random", rand_ok),
            ("v6d2 > wrong", wrong_ok),
            ("QED stable", qed_ok),
        ]
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    GEN_PARAMS["device"] = args.device
    progress = {} if args.force else load_progress()

    if args.dry_run:
        run_generation(None, None, {}, dry_run=True)
        return

    if not args.analyze_only:
        print("Loading models...")
        potential = load_esfield_potential(POTENTIAL_DEFAULT_CKPT, device=args.device)
        model = load_drugflow_model(DRUGFLOW_CKPT, device=args.device)
        print(f"DrugFlow: {sum(p.numel() for p in model.parameters()):,} params")
        t0 = time.time()
        try:
            run_generation(model, potential, progress)
        finally:
            save_progress(progress)
        print(f"Total wall time: {(time.time()-t0)/3600:.1f}h")

    analyze()

if __name__ == "__main__":
    main()
