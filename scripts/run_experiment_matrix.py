#!/usr/bin/env python3
"""Unified experiment matrix driver for ESField Steps 1-4.

Orchestrates all GPU generation runs for the four-step experiment framework.
Handles resumption, deduplication across steps, and progress tracking.

Usage:
  python scripts/run_experiment_matrix.py --step 1           # Step 1 only
  python scripts/run_experiment_matrix.py --step 2           # Step 2 only
  python scripts/run_experiment_matrix.py --step 3           # Step 3 only
  python scripts/run_experiment_matrix.py --step 4           # Step 4 only
  python scripts/run_experiment_matrix.py --step all         # All steps
  python scripts/run_experiment_matrix.py --dry-run          # Show plan without running
  python scripts/run_experiment_matrix.py --step 1 --pockets 2jkr,1sle  # Specific pockets

Output structure:
  experiments/pdbbind_water_sites/experiment_matrix/
    step1/{pocket_id}_baseline/       # 20 pockets × baseline
    step2/{pocket_id}_{condition}/     #  5 pockets × 3 conditions
    step3/{pocket_id}_{condition}/     # 20 pockets × 4 conditions
    step4/{pocket_id}_lambda{val}/     # 10 pockets × 8 λ values
    progress.json
"""

from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drugflow_esfield_guide import (
    load_esfield_potential, load_drugflow_model, generate_molecules,
    DRUGFLOW_CKPT,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POTENTIAL_CKPT = str(ROOT / "experiments/potential_training/train_gpu/compatibility_potential_epoch_0200.pt")
TEST_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"
SITE_MAPS_DIR = ROOT / "experiments/pdbbind_water_sites/test_sites"
OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/experiment_matrix"

# Step 2 ablation pockets — diverse in site type composition
ABLATION_POCKETS = ["5g60", "3b28", "6ayn", "3r01", "1sle"]

# Step 4 λ sweep (8 values including baseline control)
LAMBDA_SWEEP = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# Shared generation parameters
GEN_PARAMS = {
    "num_samples": 20,
    "timesteps": 40,
    "gen_batch_size": 5,
    "guidance_start": 0.4,
    "guidance_end": 0.85,
    "device": "cuda:0",
}

EXPECTED_STEP1 = 20   # pockets
EXPECTED_STEP2 = 5    # pockets × 3 conditions = 15 runs
EXPECTED_STEP3 = 20   # pockets × 3 conditions (+1 reused) = 60 runs
EXPECTED_STEP4 = 10   # pockets × 8 λ = 80 runs


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress():
    path = OUTPUT_BASE / "progress.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"step1": {}, "step2": {}, "step3": {}, "step4": {}, "meta": {}}


def save_progress(progress):
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    progress["meta"]["last_updated"] = datetime.now().isoformat()
    with open(OUTPUT_BASE / "progress.json", "w") as f:
        json.dump(progress, f, indent=2, default=str)


def run_key(pocket_id, condition):
    """Unique key for a run. Used both as directory name and progress key."""
    return f"{pocket_id}_{condition}"


def is_done(progress, step, pocket_id, condition):
    return progress.get(step, {}).get(run_key(pocket_id, condition)) == "done"


def mark_done(progress, step, pocket_id, condition):
    progress.setdefault(step, {})[run_key(pocket_id, condition)] = "done"
    save_progress(progress)


def run_exists_and_valid(pocket_id, condition, step_dir):
    """Check if a previous run exists with valid output."""
    run_dir = OUTPUT_BASE / step_dir / run_key(pocket_id, condition)
    sdf = run_dir / "molecules.sdf"
    meta = run_dir / "metadata.json"
    if sdf.exists() and meta.exists():
        try:
            m = json.loads(meta.read_text())
            if m.get("metrics", {}).get("valid", 0) > 0:
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Pocket data helpers
# ---------------------------------------------------------------------------

def load_test_pockets():
    return json.loads(TEST_POCKETS_JSON.read_text())


def get_site_map_path(pocket_id, condition="correct"):
    """Get path to a site map for a pocket and condition."""
    return SITE_MAPS_DIR / condition / f"{pocket_id}_site_map.json"


def get_pocket_paths(pocket_info):
    """Resolve protein PDB, pocket PDB, and ligand SDF paths."""
    pocket_dir = Path(pocket_info["dir"])
    pdb_id = pocket_info["pdb_id"]
    return {
        "protein_pdb": str(pocket_dir / f"{pdb_id}_protein.pdb"),
        "pocket_pdb": str(pocket_dir / f"{pdb_id}_pocket.pdb"),
        "ref_ligand": str(pocket_dir / f"{pdb_id}_ligand.sdf"),
        "pdb_id": pdb_id,
    }


# ---------------------------------------------------------------------------
# Step 1: Baseline generation (20 pockets × λ=0)
# ---------------------------------------------------------------------------

def run_step1(model, potential, progress, pockets=None, dry_run=False):
    """Generate baseline (unguided) molecules for all test pockets."""
    step_dir = "step1"
    test_pockets = load_test_pockets()
    if pockets:
        test_pockets = [p for p in test_pockets if p["pdb_id"] in pockets]

    runs = []
    for p in test_pockets:
        paths = get_pocket_paths(p)
        condition = "baseline"
        if is_done(progress, "step1", paths["pdb_id"], condition):
            continue
        if run_exists_and_valid(paths["pdb_id"], condition, step_dir):
            mark_done(progress, "step1", paths["pdb_id"], condition)
            continue
        runs.append((paths, condition, 0.0, None))  # None site_map for baseline

    if dry_run:
        print(f"\n[Step 1] Would run {len(runs)} baseline generations "
              f"({len(runs) * GEN_PARAMS['num_samples']} molecules)")
        for paths, cond, lam, sm in runs:
            print(f"  {paths['pdb_id']}: baseline (λ=0)")
        return

    if not runs:
        print("[Step 1] All baseline runs complete.")
        return

    print(f"\n{'='*60}")
    print(f"[Step 1] Generating baseline for {len(runs)} pockets")
    print(f"{'='*60}")

    for i, (paths, condition, lam, site_map_path) in enumerate(runs):
        pid = paths["pdb_id"]
        print(f"\n[{i+1}/{len(runs)}] {pid} baseline (λ=0)")
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=get_site_map_path(pid, "correct"),  # needed for structure, not used for guidance
                output_dir=str(OUTPUT_BASE / step_dir / run_key(pid, condition)),
                esfield_lambda=0.0,
                **GEN_PARAMS,
            )
            print(f"  Valid: {result['valid_count']}/{GEN_PARAMS['num_samples']}, "
                  f"QED: {result['metrics']['qed_mean']:.3f}, "
                  f"{result['elapsed']:.0f}s")
            mark_done(progress, "step1", pid, condition)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    save_progress(progress)


# ---------------------------------------------------------------------------
# Step 2: Site physical meaningfulness (5 pockets × 3 conditions)
# ---------------------------------------------------------------------------

def run_step2(model, potential, progress, pockets=None, dry_run=False):
    """Guided generation with correct/random/shuffled site maps."""
    step_dir = "step2"
    test_pockets = load_test_pockets()
    pocket_ids = pockets or ABLATION_POCKETS
    test_pockets = [p for p in test_pockets if p["pdb_id"] in pocket_ids]

    conditions = ["correct", "random", "shuffled"]
    runs = []
    for p in test_pockets:
        paths = get_pocket_paths(p)
        pid = paths["pdb_id"]
        for cond in conditions:
            if is_done(progress, "step2", pid, cond):
                continue
            if run_exists_and_valid(pid, cond, step_dir):
                mark_done(progress, "step2", pid, cond)
                continue
            site_map_path = str(get_site_map_path(pid, cond))
            runs.append((paths, cond, 1.0, site_map_path))

    if dry_run:
        print(f"\n[Step 2] Would run {len(runs)} guided generations "
              f"({len(runs) * GEN_PARAMS['num_samples']} molecules)")
        for paths, cond, lam, sm in runs:
            print(f"  {paths['pdb_id']}: {cond} (λ=1.0)")
        return

    if not runs:
        print("[Step 2] All site meaningfulness runs complete.")
        return

    print(f"\n{'='*60}")
    print(f"[Step 2] Site meaningfulness: {len(runs)} runs")
    print(f"{'='*60}")

    for i, (paths, condition, lam, site_map_path) in enumerate(runs):
        pid = paths["pdb_id"]
        print(f"\n[{i+1}/{len(runs)}] {pid} {condition} (λ={lam})")
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=site_map_path,
                output_dir=str(OUTPUT_BASE / step_dir / run_key(pid, condition)),
                esfield_lambda=lam,
                **GEN_PARAMS,
            )
            print(f"  Valid: {result['valid_count']}/{GEN_PARAMS['num_samples']}, "
                  f"QED: {result['metrics']['qed_mean']:.3f}, "
                  f"{result['elapsed']:.0f}s")
            mark_done(progress, "step2", pid, condition)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    save_progress(progress)


# ---------------------------------------------------------------------------
# Step 3: ESField improvement (20 pockets × 4 conditions)
# ---------------------------------------------------------------------------

def run_step3(model, potential, progress, pockets=None, dry_run=False):
    """Compare baseline vs λ=0.5 vs λ=1.0 vs random site map.

    Baseline is reused from Step 1. Random from Step 2 is reused for
    overlapping ablation pockets.
    """
    step_dir = "step3"
    test_pockets = load_test_pockets()
    if pockets:
        test_pockets = [p for p in test_pockets if p["pdb_id"] in pockets]

    conditions = [
        ("lambda0.5", 0.5, "correct"),
        ("lambda1.0", 1.0, "correct"),
        ("random", 1.0, "random"),
    ]

    runs = []
    for p in test_pockets:
        paths = get_pocket_paths(p)
        pid = paths["pdb_id"]

        # Baseline: symlink from step1
        baseline_src = OUTPUT_BASE / "step1" / run_key(pid, "baseline")
        baseline_dst = OUTPUT_BASE / step_dir / run_key(pid, "baseline")
        if baseline_src.exists() and not baseline_dst.exists():
            baseline_dst.parent.mkdir(parents=True, exist_ok=True)
            baseline_dst.symlink_to(baseline_src.resolve(), target_is_directory=True)
            mark_done(progress, "step3", pid, "baseline")

        for cond_name, lam, site_cond in conditions:
            if is_done(progress, "step3", pid, cond_name):
                continue
            if run_exists_and_valid(pid, cond_name, step_dir):
                mark_done(progress, "step3", pid, cond_name)
                continue

            site_map_path = str(get_site_map_path(pid, site_cond))

            # Check if this exact run exists in Step 2 (random condition for ablation pockets)
            if site_cond == "random" and pid in ABLATION_POCKETS:
                step2_src = OUTPUT_BASE / "step2" / run_key(pid, "random")
                step3_dst = OUTPUT_BASE / step_dir / run_key(pid, cond_name)
                if step2_src.exists():
                    step3_dst.parent.mkdir(parents=True, exist_ok=True)
                    if not step3_dst.exists():
                        step3_dst.symlink_to(step2_src.resolve(), target_is_directory=True)
                    mark_done(progress, "step3", pid, cond_name)
                    continue

            runs.append((paths, cond_name, lam, site_map_path))

    if dry_run:
        print(f"\n[Step 3] Would run {len(runs)} guided generations "
              f"({len(runs) * GEN_PARAMS['num_samples']} molecules)")
        for paths, cond, lam, sm in runs[:10]:
            print(f"  {paths['pdb_id']}: {cond} (λ={lam})")
        if len(runs) > 10:
            print(f"  ... and {len(runs) - 10} more")
        return

    if not runs:
        print("[Step 3] All improvement runs complete (including reused baselines).")
        return

    print(f"\n{'='*60}")
    print(f"[Step 3] ESField improvement: {len(runs)} runs")
    print(f"{'='*60}")

    for i, (paths, condition, lam, site_map_path) in enumerate(runs):
        pid = paths["pdb_id"]
        print(f"\n[{i+1}/{len(runs)}] {pid} {condition} (λ={lam})")
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=site_map_path,
                output_dir=str(OUTPUT_BASE / step_dir / run_key(pid, condition)),
                esfield_lambda=lam,
                **GEN_PARAMS,
            )
            print(f"  Valid: {result['valid_count']}/{GEN_PARAMS['num_samples']}, "
                  f"QED: {result['metrics']['qed_mean']:.3f}, "
                  f"{result['elapsed']:.0f}s")
            mark_done(progress, "step3", pid, condition)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    save_progress(progress)


# ---------------------------------------------------------------------------
# Step 4: Quality-constrained validation (10 pockets × 8 λ values)
# ---------------------------------------------------------------------------

def run_step4(model, potential, progress, pockets=None, dry_run=False):
    """λ sweep across 10 pockets to find Pareto frontier."""
    step_dir = "step4"
    test_pockets = load_test_pockets()

    # Use first 10 test pockets as default for λ sweep
    step4_pockets = pockets or [p["pdb_id"] for p in test_pockets[:10]]
    test_pockets = [p for p in test_pockets if p["pdb_id"] in step4_pockets]

    runs = []
    for p in test_pockets:
        paths = get_pocket_paths(p)
        pid = paths["pdb_id"]
        for lam in LAMBDA_SWEEP:
            cond_name = f"lambda{lam}"
            if is_done(progress, "step4", pid, cond_name):
                continue
            if run_exists_and_valid(pid, cond_name, step_dir):
                mark_done(progress, "step4", pid, cond_name)
                continue

            site_map_path = str(get_site_map_path(pid, "correct"))
            runs.append((paths, cond_name, lam, site_map_path))

    if dry_run:
        print(f"\n[Step 4] Would run {len(runs)} λ sweep generations "
              f"({len(runs) * GEN_PARAMS['num_samples']} molecules)")
        for lam in LAMBDA_SWEEP:
            n = sum(1 for _, cond, _, _ in runs if cond == f"lambda{lam}")
            if n:
                print(f"  λ={lam}: {n} pockets")
        return

    if not runs:
        print("[Step 4] All quality sweep runs complete.")
        return

    print(f"\n{'='*60}")
    print(f"[Step 4] Quality-constrained λ sweep: {len(runs)} runs")
    print(f"{'='*60}")

    for i, (paths, condition, lam, site_map_path) in enumerate(runs):
        pid = paths["pdb_id"]
        print(f"\n[{i+1}/{len(runs)}] {pid} {condition}")
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=site_map_path,
                output_dir=str(OUTPUT_BASE / step_dir / run_key(pid, condition)),
                esfield_lambda=lam,
                **GEN_PARAMS,
            )
            print(f"  Valid: {result['valid_count']}/{GEN_PARAMS['num_samples']}, "
                  f"QED: {result['metrics']['qed_mean']:.3f}, "
                  f"{result['elapsed']:.0f}s")
            mark_done(progress, "step4", pid, condition)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    save_progress(progress)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(progress):
    """Print a summary of experiment progress."""
    steps = {
        "step1": "Baseline diagnosis (20 pockets × λ=0)",
        "step2": "Site meaningfulness (5 pockets × 3 conditions)",
        "step3": "ESField improvement (20 pockets × 4 conditions)",
        "step4": "Quality constraint (10 pockets × 8 λ)",
    }
    print(f"\n{'='*60}")
    print("Experiment Matrix Status")
    print(f"{'='*60}")
    for step, desc in steps.items():
        done = sum(1 for v in progress.get(step, {}).values() if v == "done")
        expected = {"step1": EXPECTED_STEP1, "step2": EXPECTED_STEP2 * 3,
                    "step3": EXPECTED_STEP3 * 4, "step4": EXPECTED_STEP4 * 8}
        exp = expected.get(step, 1)
        bar = "█" * int(done / max(exp, 1) * 20)
        print(f"  {step}: [{bar:<20}] {done}/{exp}  {desc}")


def estimate_gpu_time(progress):
    """Estimate remaining GPU time."""
    expected = {
        "step1": EXPECTED_STEP1,
        "step2": EXPECTED_STEP2 * 3,
        "step3": EXPECTED_STEP3 * 3,   # excluding baseline (reused)
        "step4": EXPECTED_STEP4 * 8,
    }
    total_remaining = 0
    for step, exp in expected.items():
        done = sum(1 for v in progress.get(step, {}).values() if v == "done")
        remaining = max(0, exp - done)
        total_remaining += remaining

    sec_per_sample = 4.5  # conservative
    samples_per_run = GEN_PARAMS["num_samples"]
    total_seconds = total_remaining * samples_per_run * sec_per_sample
    hours = total_seconds / 3600
    print(f"\nEstimated remaining: {total_remaining} runs × {samples_per_run} samples "
          f"= {total_remaining * samples_per_run} generations")
    print(f"Estimated GPU time: {hours:.1f}h (A100, {sec_per_sample}s/sample)")
    print(f"Estimated Vina docking: {total_remaining * samples_per_run * 15 / 3600:.1f}h (CPU, parallelizable)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ESField experiment matrix driver")
    parser.add_argument("--step", default="all",
                        choices=["1", "2", "3", "4", "all"],
                        help="Which step(s) to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without generating")
    parser.add_argument("--pockets", default=None,
                        help="Comma-separated pocket IDs (default: all for the step)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Molecules per GPU batch")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip completed runs (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if already done")
    args = parser.parse_args()

    # Update global params
    GEN_PARAMS["device"] = args.device
    GEN_PARAMS["gen_batch_size"] = args.batch_size

    # Load progress
    progress = {} if args.force else load_progress()
    if args.force:
        print("[force mode] Ignoring previous progress.")

    pockets = args.pockets.split(",") if args.pockets else None

    # Dry run
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no molecules will be generated")
        print("=" * 60)
        print(f"Test pockets: {len(load_test_pockets())}")
        print(f"Samples per run: {GEN_PARAMS['num_samples']}")
        print(f"GPU: {args.device}, batch size: {args.batch_size}")
        print(f"Potential: {POTENTIAL_CKPT}")
        print(f"DrugFlow: {DRUGFLOW_CKPT}")

        # Need mock model for dry run (we don't load models in dry run)
        model = potential = None
        if args.step in ("1", "all"):
            run_step1(model, potential, progress, pockets, dry_run=True)
        if args.step in ("2", "all"):
            run_step2(model, potential, progress, pockets, dry_run=True)
        if args.step in ("3", "all"):
            run_step3(model, potential, progress, pockets, dry_run=True)
        if args.step in ("4", "all"):
            run_step4(model, potential, progress, pockets, dry_run=True)

        estimate_gpu_time(progress)
        return

    # Load models once
    print("Loading models...")
    potential = load_esfield_potential(POTENTIAL_CKPT, device=args.device)
    print(f"  Potential loaded ({sum(p.numel() for p in potential.parameters()):,} params)")
    model = load_drugflow_model(DRUGFLOW_CKPT, device=args.device)
    print(f"  DrugFlow loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    t0 = time.time()

    try:
        if args.step in ("1", "all"):
            run_step1(model, potential, progress, pockets)
        if args.step in ("2", "all"):
            run_step2(model, potential, progress, pockets)
        if args.step in ("3", "all"):
            run_step3(model, potential, progress, pockets)
        if args.step in ("4", "all"):
            run_step4(model, potential, progress, pockets)
    finally:
        save_progress(progress)

    total_elapsed = time.time() - t0
    print(f"\nTotal wall time: {total_elapsed/3600:.1f}h")
    print_summary(progress)


if __name__ == "__main__":
    main()
