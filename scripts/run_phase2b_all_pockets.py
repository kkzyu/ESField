#!/usr/bin/env python3
"""Phase IIb — Full Docking with MMFF94 Minimization for all 10 pockets.

Pipeline: SDF → MMFF94 minimize (batch) → Vina full docking → energy trend analysis.

Key difference from Phase II: full conformational search (not --score_only) on
pre-minimized molecules yields meaningful Vina scores (-10 to +10 kcal/mol range).

Usage:
    python scripts/run_phase2b_all_pockets.py --pocket 2gni --n-workers 4
    python scripts/run_phase2b_all_pockets.py --all --n-workers 4
"""

import argparse, csv, json, glob, os, subprocess, sys, time
from pathlib import Path

ESFIELD_ROOT = str(Path(__file__).resolve().parents[1])
SCRIPT_DIR = os.path.join(ESFIELD_ROOT, "scripts")
PDB_ROOT = "/root/autodl-tmp/data/PDB/P-L"
SITE_DIR = os.path.join(ESFIELD_ROOT, "experiments/pdbbind_water_sites/site_maps")
GEN_DIR = os.path.join(ESFIELD_ROOT, "experiments/pdbbind_water_sites/v71_full_study")

sys.path.insert(0, os.path.join(ESFIELD_ROOT, "src"))


def run_cmd(cmd, desc=""):
    print(f"  [{desc}] Running...")
    return subprocess.run(cmd, capture_output=False, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", default=None, help="Single pocket")
    parser.add_argument("--all", action="store_true", help="All 10 pockets")
    parser.add_argument("--pocket-list", default=f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_ablation/pocket_ids_10.txt")
    parser.add_argument("--out-dir", default=f"{ESFIELD_ROOT}/experiments/phase2b_results")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--mode", default="full_dock", choices=["full_dock", "score_only"])
    parser.add_argument("--skip-minimize", action="store_true")
    parser.add_argument("--skip-docking", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pocket:
        pockets = [args.pocket]
    elif args.all:
        with open(args.pocket_list) as f:
            pockets = [l.strip() for l in f if l.strip()]
    else:
        parser.error("Use --pocket or --all")

    print("=" * 60)
    print(f"Phase IIb: {len(pockets)} pockets, mode={args.mode}")
    print(f"Minimize: {not args.skip_minimize}, Dock: {not args.skip_docking}")
    print("=" * 60)

    all_results = []
    t0 = time.time()

    for i, pocket in enumerate(pockets):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(pockets)}] {pocket}")
        print(f"{'='*60}")

        sdf_input = os.path.join(GEN_DIR, f"{pocket}_v7.1.sdf")
        if not os.path.exists(sdf_input):
            print(f"  SKIP: SDF not found")
            continue

        pdirs = glob.glob(os.path.join(PDB_ROOT, "*", pocket))
        if not pdirs:
            print(f"  SKIP: protein not found")
            continue
        pdir = pdirs[0]
        protein_pdb = os.path.join(pdir, f"{pocket}_protein.pdb")
        pocket_pdb = os.path.join(pdir, f"{pocket}_pocket.pdb")
        site_map_path = os.path.join(SITE_DIR, f"{pocket}_site_map.json")

        # Step 1: Batch minimize SDF
        sdf_minimized = os.path.join(args.out_dir, f"{pocket}_minimized.sdf")
        if not args.skip_minimize:
            print(f"  [1/3] MMFF94 Minimization")
            from utils.minimize_molecule import batch_minimize_sdf
            summary = batch_minimize_sdf(
                sdf_input, sdf_minimized, max_iters=200,
                force_tolerance=0.01, n_jobs=args.n_workers,
            )
            print(f"    {summary['n_success']}/{summary['n_input']} minimized, "
                  f"energy: {summary['mean_energy_before']:.1f}→{summary['mean_energy_after']:.1f}")
        else:
            sdf_minimized = sdf_input
            print(f"  [1/3] Minimization SKIPPED")

        if not os.path.exists(sdf_minimized):
            print(f"  SKIP: minimization failed")
            continue

        # Step 2: Vina full docking
        docking_csv = os.path.join(args.out_dir, f"{pocket}_docking.csv")
        if not args.skip_docking:
            print(f"  [2/3] Vina Docking ({args.mode})")
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "compute_vina_docking.py"),
                "--pocket-id", pocket,
                "--sdf-file", sdf_minimized,
                "--protein-pdb", protein_pdb,
                "--pocket-pdb", pocket_pdb,
                "--output-csv", docking_csv,
                "--mode", args.mode,
                "--n-workers", str(args.n_workers),
                "--pre-minimized",
            ]
            run_cmd(cmd, f"Vina docking {pocket}")
        else:
            print(f"  [2/3] Docking SKIPPED")

        if not os.path.exists(docking_csv):
            print(f"  WARNING: docking failed, skipping analysis")
            continue

        # Step 3: Energy trend
        print(f"  [3/3] Energy Trend Analysis")
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "analyze_energy_trend.py"),
            "--pocket-id", pocket,
            "--docking-csv", docking_csv,
            "--sdf-file", sdf_minimized,
            "--occupancy-sdf", sdf_input,  # Use original SDF for occupancy labels
            "--site-map", site_map_path,
            "--output-dir", args.out_dir,
        ]
        run_cmd(cmd, f"Energy trend {pocket}")

        # Load result
        result_json = os.path.join(args.out_dir, f"{pocket}_energy_trend.json")
        if os.path.exists(result_json):
            result = json.load(open(result_json))
            all_results.append(result)
            print(f"    Occ={result['n_occ']}, NonOcc={result['n_nonocc']}, "
                  f"p={result.get('p_value', 'N/A')}, "
                  f"MeanOcc={result.get('mean_score_occ', 'N/A')}, "
                  f"MeanNon={result.get('mean_score_nonocc', 'N/A')}")

    total_elapsed = time.time() - t0

    # Summary
    if all_results:
        csv_path = os.path.join(args.out_dir, "phase2b_summary.csv")
        fields = ["pocket", "n_occ", "n_nonocc", "mean_score_occ", "mean_score_nonocc",
                  "median_score_occ", "median_score_nonocc", "p_value", "cliff_delta",
                  "significance"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in all_results:
                w.writerow(r)

        print(f"\n{'='*70}")
        print(f"PHASE IIb SUMMARY ({args.mode})")
        print(f"{'='*70}")
        header = (f"{'Pocket':<8} {'Occ':>4} {'NonO':>4} {'MeanOcc':>8} {'MeanNon':>8} "
                  f"{'p':>7} {'Cliff δ':>8} {'Sig':>4}")
        print(header)
        print("-" * len(header))
        for r in all_results:
            pv = f"{r['p_value']:.3f}" if r.get('p_value') is not None else "--"
            mo = f"{r['mean_score_occ']:.1f}" if r.get('mean_score_occ') is not None else "--"
            mn = f"{r['mean_score_nonocc']:.1f}" if r.get('mean_score_nonocc') is not None else "--"
            cd = f"{r['cliff_delta']:.3f}" if r.get('cliff_delta') is not None else "--"
            print(f"{r['pocket']:<8} {r['n_occ']:>4} {r['n_nonocc']:>4} "
                  f"{mo:>8} {mn:>8} {pv:>7} {cd:>8} {r.get('significance','--'):>4}")

    print(f"\nTotal: {total_elapsed:.1f}s → {args.out_dir}/")


if __name__ == "__main__":
    main()
