#!/usr/bin/env python3
"""Phase II batch runner — Vina scoring + energy trend analysis for all pockets.

Usage:
    PYTHONPATH=src python scripts/run_phase2_all_pockets.py \
        --pocket-list experiments/v71_ablation/pocket_ids_10.txt \
        --gen-dir experiments/v71_full_study \
        --out-dir experiments/phase2_results
"""

import argparse, csv, json, glob, os, subprocess, sys, time
from pathlib import Path

ESFIELD_ROOT = str(Path(__file__).resolve().parents[1])
SCRIPT_DIR = os.path.join(ESFIELD_ROOT, "scripts")
PDB_ROOT = "/root/autodl-tmp/data/PDB/P-L"
SITE_DIR = os.path.join(ESFIELD_ROOT, "experiments/pdbbind_water_sites/site_maps")

sys.path.insert(0, os.path.join(ESFIELD_ROOT, "src"))


def run_cmd(cmd, desc=""):
    print(f"  [{desc}] {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-200:]}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-list", default=None)
    parser.add_argument("--gen-dir", default=f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_full_study")
    parser.add_argument("--out-dir", default=f"{ESFIELD_ROOT}/experiments/phase2_results")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--skip-scoring", action="store_true", help="Skip Vina scoring (if already done)")
    parser.add_argument("--pocket", default=None, help="Run single pocket only")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load pocket list
    if args.pocket:
        pockets = [args.pocket]
    elif args.pocket_list:
        with open(args.pocket_list) as f:
            pockets = [l.strip() for l in f if l.strip()]
    else:
        parser.error("Either --pocket or --pocket-list is required")
    print(f"Phase II: {len(pockets)} pockets")
    print(f"Gen dir: {args.gen_dir}")
    print(f"Output: {args.out_dir}")

    all_results = []
    t0 = time.time()

    for i, pocket in enumerate(pockets):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(pockets)}] {pocket}")
        print(f"{'='*60}")

        sdf_file = os.path.join(args.gen_dir, f"{pocket}_v7.1.sdf")
        if not os.path.exists(sdf_file):
            print(f"  SKIP: SDF not found: {sdf_file}")
            continue

        # Find protein
        pdirs = glob.glob(os.path.join(PDB_ROOT, "*", pocket))
        if not pdirs:
            print(f"  SKIP: protein dir not found")
            continue
        pdir = pdirs[0]
        protein_pdb = os.path.join(pdir, f"{pocket}_protein.pdb")
        pocket_pdb = os.path.join(pdir, f"{pocket}_pocket.pdb")
        site_map = os.path.join(SITE_DIR, f"{pocket}_site_map.json")

        if not os.path.exists(site_map):
            print(f"  SKIP: site map not found")
            continue

        # Step 1: Vina scoring
        vina_csv = os.path.join(args.out_dir, f"{pocket}_vina_scores.csv")
        if not args.skip_scoring or not os.path.exists(vina_csv):
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "compute_vina_scores.py"),
                "--pocket-id", pocket,
                "--sdf-file", sdf_file,
                "--protein-pdb", protein_pdb,
                "--pocket-pdb", pocket_pdb,
                "--output-csv", vina_csv,
                "--n-workers", str(args.n_workers),
            ]
            run_cmd(cmd, f"Vina scoring {pocket}")
        else:
            print(f"  Vina scoring: SKIPPED (already done)")

        if not os.path.exists(vina_csv):
            print(f"  WARNING: Vina scoring failed, skipping analysis")
            continue

        # Step 2: Energy trend analysis
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "analyze_energy_trend.py"),
            "--pocket-id", pocket,
            "--docking-csv", vina_csv,
            "--sdf-file", sdf_file,
            "--site-map", site_map,
            "--output-dir", args.out_dir,
        ]
        run_cmd(cmd, f"Energy trend {pocket}")

        # Load result
        result_json = os.path.join(args.out_dir, f"{pocket}_energy_trend.json")
        if os.path.exists(result_json):
            result = json.load(open(result_json))
            all_results.append(result)

    total_elapsed = time.time() - t0

    # ── Summary CSV ──
    if all_results:
        summary_csv = os.path.join(args.out_dir, "phase2_summary.csv")
        fields = ["pocket", "n_occ", "n_nonocc", "mean_score_occ", "mean_score_nonocc",
                  "median_score_occ", "median_score_nonocc", "p_value", "cliff_delta",
                  "cohens_d", "pct_occ_better_than_nonocc_median", "significance"]
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        # Print summary
        print(f"\n{'='*70}")
        print("PHASE II SUMMARY — Vina Energy Trend")
        print(f"{'='*70}")
        header = (f"{'Pocket':<8} {'Occ':>4} {'NonO':>4} {'MeanOcc':>8} {'MeanNon':>8} "
                  f"{'p':>7} {'Cliff δ':>8} {'%Better':>8} {'Sig':>4}")
        print(header)
        print("-" * len(header))
        for r in all_results:
            if r.get("p_value") is None:
                print(f"{r['pocket']:<8} {r['n_occ']:>4} {r['n_nonocc']:>4} "
                      f"{'--':>8} {'--':>8} {'--':>7} {'--':>8} {'--':>8} {'insuf':>4}")
            else:
                print(f"{r['pocket']:<8} {r['n_occ']:>4} {r['n_nonocc']:>4} "
                      f"{r['mean_score_occ']:>8.1f} {r['mean_score_nonocc']:>8.1f} "
                      f"{r['p_value']:>7.3f} {r['cliff_delta']:>8.3f} "
                      f"{r['pct_occ_better_than_nonocc_median']:>7.1%} "
                      f"{r['significance']:>4}")

        # ── LaTeX table ──
        latex_path = os.path.join(args.out_dir, "phase2_summary_table.tex")
        with open(latex_path, "w") as f:
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{Vina docking scores: occupied vs.\ non-occupied molecules. "
                    r"Negative Cliff's $\delta$ = occupied scores are lower (better). "
                    r"Mann-Whitney U test.}" + "\n")
            f.write(r"\label{tab:phase2_energy}" + "\n")
            f.write(r"\begin{tabular}{lrrrrrcr}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Pocket & $N_{\text{occ}}$ & $N_{\text{non}}$ & "
                    r"$\bar{E}_{\text{occ}}$ & $\bar{E}_{\text{non}}$ & "
                    r"Cliff's $\delta$ & $p$ & Sig. \\" + "\n")
            f.write(r"\midrule" + "\n")
            for r in all_results:
                if r.get("p_value") is None:
                    f.write(f"{r['pocket']} & {r['n_occ']} & {r['n_nonocc']} & "
                            f"— & — & — & — & insuf. \\\\\n")
                else:
                    pv = f"{r['p_value']:.3f}" if r['p_value'] >= 0.001 else r"$<\!0.001$"
                    f.write(f"{r['pocket']} & {r['n_occ']} & {r['n_nonocc']} & "
                            f"{r['mean_score_occ']:.1f} & {r['mean_score_nonocc']:.1f} & "
                            f"{r['cliff_delta']:.3f} & {pv} & "
                            f"{r['significance']} \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")
        print(f"\nLaTeX table saved to {latex_path}")

    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Results: {args.out_dir}/")


if __name__ == "__main__":
    main()
