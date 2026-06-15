#!/usr/bin/env python3
"""
Task 3: TargetDiff 6-Pocket Cross-Architecture Validation.

Extends run_targetdiff_full_pipeline.py to all 6 PDBbind pockets:
  3mfw, 2gni, 6o4x, 2jke, 2gqn, 6phx

Three conditions per pocket:
  - unguided:  Raw TargetDiff DDPM sampling
  - hard_fix:  Hard-overwrite anchor atoms (expected: 0% validity)
  - kinematic: CoM-only guidance (Ours, expected: high validity + low strain)

Key metrics: DirectOcc_HEW, DirectOcc_SW, Validity, Vina, QED, SA, Strain/atom.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

# ── Pocket config (extended to 6 pockets) ──
POCKET_CONFIG = {
    "3mfw": {"year": "2001-2010", "n_hew": 7, "ref_atoms": 26},
    "2gni": {"year": "2001-2010", "n_hew": 3, "ref_atoms": 20},
    "6o4x": {"year": "2011-2019", "n_hew": 6, "ref_atoms": 22},
    "2jke": {"year": "2001-2010", "n_hew": 4, "ref_atoms": 24},
    "2gqn": {"year": "2001-2010", "n_hew": 7, "ref_atoms": 18},
    "6phx": {"year": "2011-2019", "n_hew": 5, "ref_atoms": 21},
}

CONDITIONS = ["unguided", "hard_fix", "kinematic"]


def run_targetdiff_pocket(pocket, condition, n_samples=50, n_steps=1000,
                          output_dir=None, device="cuda", timeout=7200):
    """Run TargetDiff generation for one pocket × condition.

    Uses the existing run_targetdiff_full_pipeline.py via subprocess.
    """
    out_dir = Path(output_dir) / pocket / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    script = ROOT / "scripts" / "run_targetdiff_full_pipeline.py"
    cmd = [
        sys.executable, str(script),
        "--pocket", pocket,
        "--mode", condition,
        "--n-samples", str(n_samples),
        "--num-steps", str(n_steps),
        "--output-dir", str(out_dir),
        "--device", device,
    ]

    print(f"  [{pocket}/{condition}] Starting (N={n_samples}, T={n_steps})...")
    t0 = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, cwd=str(ROOT))
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [{pocket}/{condition}] FAILED (rc={result.returncode})")
            print(f"    STDERR: {result.stderr[-300:]}")
            return {"status": "failed", "rc": result.returncode,
                    "stderr": result.stderr[-1000:]}

        print(f"  [{pocket}/{condition}] Done in {elapsed:.0f}s")
        return {"status": "completed", "elapsed_s": elapsed}

    except subprocess.TimeoutExpired:
        print(f"  [{pocket}/{condition}] TIMEOUT ({timeout}s)")
        return {"status": "timeout"}


def collect_metrics(pocket, condition, output_dir):
    """Collect and aggregate metrics from generated SDFs + evaluation JSONs."""
    out_dir = Path(output_dir) / pocket / condition

    # Try to read existing summary JSON
    summary_json = out_dir / "summary.json"
    if summary_json.exists():
        with open(summary_json) as f:
            existing = json.load(f)
    else:
        existing = {}

    # Run evaluator on SDFs
    sdf_dir = out_dir / "sdfs" if (out_dir / "sdfs").exists() else out_dir
    sdf_files = list(sdf_dir.glob("*.sdf")) if sdf_dir.exists() else []

    metrics = {"pocket": pocket, "condition": condition,
               "n_sdf_files": len(sdf_files), **existing}

    if sdf_files:
        # Combine and evaluate
        combined = out_dir / "_combined.sdf"
        with open(combined, "w") as out:
            for f in sorted(sdf_files):
                with open(f) as inp:
                    d = inp.read()
                    if d.strip(): out.write(d)

        try:
            from evaluator import (
                compute_validity_batch, compute_strain_energy_batch,
                compute_qed_batch, compute_sa_score_batch,
                compute_site_occupancy_batch,
            )

            v = compute_validity_batch(combined)
            metrics["validity"] = v

            se = compute_strain_energy_batch(combined)
            metrics["strain_per_atom_mean"] = se.get("strain_per_atom_mean")
            metrics["strain_per_atom_std"] = se.get("strain_per_atom_std")

            qe = compute_qed_batch(combined)
            metrics["qed_mean"] = qe.get("qed_mean")
            metrics["qed_std"] = qe.get("qed_std")

            sa = compute_sa_score_batch(combined)
            metrics["sa_mean"] = sa.get("sa_score_mean")

            # Site occupancy
            sm = ROOT / "experiments/targetdiff_replication/site_maps" / f"{pocket}_site_map.json"
            if sm.exists():
                occ = compute_site_occupancy_batch(combined, str(sm))
                metrics["direct_occ_hew"] = occ.get("direct_occ_hew")
                metrics["direct_occ_sw"] = occ.get("direct_occ_sw")
        except Exception as e:
            metrics["eval_error"] = str(e)[:200]

    # Save
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def print_summary_table(all_results):
    """Print a summary table of all results."""
    print(f"\n{'='*90}")
    print("  TARGETDIFF 6-POCKET CROSS-VALIDATION SUMMARY")
    print(f"{'='*90}")
    header = f"  {'Pocket':<8} {'Condition':<12} {'Status':<12} {'Valid':<10} {'Strain/at':<10} {'QED':<8} {'HEW Occ':<10} {'SW Occ':<10}"
    print(header)
    print("  " + "-" * 88)
    for pocket, conds in sorted(all_results.items()):
        for cond, metrics in sorted(conds.items()):
            status = metrics.get("status", "?")[:12]
            v = metrics.get("validity", {})
            vr = v.get("validity_rate", None) if isinstance(v, dict) else None
            vr_str = f"{vr:.1%}" if vr is not None else "--"
            se = metrics.get("strain_per_atom_mean")
            se_str = f"{se:.2f}" if se is not None else "--"
            qe = metrics.get("qed_mean")
            qe_str = f"{qe:.3f}" if qe is not None else "--"
            hew = metrics.get("direct_occ_hew")
            hew_str = f"{hew:.1%}" if hew is not None else "--"
            sw = metrics.get("direct_occ_sw")
            sw_str = f"{sw:.1%}" if sw is not None else "--"
            print(f"  {pocket:<8} {cond:<12} {status:<12} {vr_str:<10} {se_str:<10} {qe_str:<8} {hew_str:<10} {sw_str:<10}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets", nargs="+", default=list(POCKET_CONFIG.keys()))
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/master_experiments/task3_targetdiff"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per condition (s)")
    parser.add_argument("--eval-only", action="store_true", help="Only collect metrics from existing runs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.eval_only:
        print("EVAL-ONLY mode: collecting metrics from existing outputs...")
        all_results = {}
        for pocket in args.pockets:
            all_results[pocket] = {}
            for cond in args.conditions:
                m = collect_metrics(pocket, cond, output_dir)
                all_results[pocket][cond] = m
        print_summary_table(all_results)
        return

    # ── Run experiments ──
    total = len(args.pockets) * len(args.conditions)
    print(f"\n{'#'*70}")
    print(f"  TARGETDIFF 6-POCKET EXPERIMENT")
    print(f"  {len(args.pockets)} pockets × {len(args.conditions)} conditions")
    print(f"  {args.n_samples} samples/condition, {args.n_steps} steps")
    print(f"  ~{total} total runs")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}\n")

    all_results = {}
    n_done = 0
    t_start = time.time()

    for pocket in args.pockets:
        all_results[pocket] = {}
        for cond in args.conditions:
            n_done += 1
            print(f"\n[{n_done}/{total}] {'='*50}")

            result = run_targetdiff_pocket(
                pocket, cond,
                n_samples=args.n_samples,
                n_steps=args.n_steps,
                output_dir=output_dir,
                device=args.device,
                timeout=args.timeout,
            )

            # Collect metrics regardless of success
            metrics = collect_metrics(pocket, cond, output_dir)
            metrics.update(result)
            all_results[pocket][cond] = metrics

            # Quick summary after each
            vr = metrics.get("validity", {})
            if isinstance(vr, dict):
                print(f"    Valid: {vr.get('n_valid','?')}/{vr.get('n_total','?')}"
                      f" = {vr.get('validity_rate',0):.1%}")
            se = metrics.get("strain_per_atom_mean")
            if se: print(f"    Strain/atom: {se:.3f}")

    elapsed = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"  ALL RUNS COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*70}")

    print_summary_table(all_results)

    # Save consolidated
    consolidated = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"n_samples": args.n_samples, "n_steps": args.n_steps,
                    "pockets": args.pockets, "conditions": args.conditions},
        "total_elapsed_s": elapsed,
        "results": all_results,
    }
    cons_path = output_dir / "consolidated_results.json"
    with open(cons_path, "w") as f:
        json.dump(consolidated, f, indent=2, default=str)
    print(f"\nConsolidated results: {cons_path}")


if __name__ == "__main__":
    main()
