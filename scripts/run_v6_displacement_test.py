#!/usr/bin/env python3
"""5-pocket HEW Displacement mechanism validation for Analytic ESField v6-D.

Tests whether v6-D analytic guidance improves HEW occupancy over baseline
and v5-learned, with proper ablation controls.

Conditions (7 total per pocket):
  baseline (lambda=0)
  v5-old-sum lambda=1.0
  v6-D lambda=0.3
  v6-D lambda=0.5
  v6-D lambda=1.0
  v6-D wrong-pocket lambda=1.0
  v6-D random-matrix lambda=1.0

Pockets: 5 (3 strong + 1 medium + 1 hard, matching v5 test)
Total: 5 pockets x 7 conditions x 20 samples = 700 generations (~0.9h GPU)

Usage:
  python scripts/run_v6_displacement_test.py --dry-run
  python scripts/run_v6_displacement_test.py
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drugflow_esfield_guide import (
    load_esfield_potential, load_drugflow_model, generate_molecules,
    POTENTIAL_DEFAULT_CKPT, DRUGFLOW_CKPT,
)

# Same 5 pockets as v5 mechanism test
MECHANISM_POCKETS = ["3ohi", "2clh", "3mfw", "4bis", "1sle"]

GEN_PARAMS = {
    "num_samples": 20,
    "timesteps": 40,
    "gen_batch_size": 5,
    "guidance_start": 0.4,
    "guidance_end": 0.85,
    "device": "cuda:0",
}

OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/v6_displacement_test"
TEST_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"
SITE_MAPS_DIR = ROOT / "experiments/pdbbind_water_sites/test_sites"

# Wrong-pocket mapping (same as v5)
WRONG_POCKET_MAP = {
    "3ohi": "5g60", "2clh": "1sle", "3mfw": "2jkr",
    "4bis": "3r01", "1sle": "2wgi",
}


def load_progress():
    path = OUTPUT_BASE / "progress.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_progress(progress):
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BASE / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def get_pocket_paths(pdb_id):
    test_pockets = json.loads(TEST_POCKETS_JSON.read_text())
    for p in test_pockets:
        if p["pdb_id"] == pdb_id:
            pocket_dir = Path(p["dir"])
            return {
                "protein_pdb": str(pocket_dir / f"{pdb_id}_protein.pdb"),
                "pocket_pdb": str(pocket_dir / f"{pdb_id}_pocket.pdb"),
                "ref_ligand": str(pocket_dir / f"{pdb_id}_ligand.sdf"),
                "pdb_id": pdb_id,
            }
    raise ValueError(f"Pocket {pdb_id} not found")


def build_run_list():
    """Build flat list of all (pid, cond_name, kwargs) tuples."""
    runs = []

    for pid in MECHANISM_POCKETS:
        # Baseline
        runs.append((pid, "baseline", {"guide_type": "learned_v5", "esfield_lambda": 0.0}))

        # v5-old-sum lambda=1.0
        runs.append((pid, "v5_old_sum_lambda1.0", {
            "guide_type": "learned_v5", "esfield_lambda": 1.0,
            "aggregation": "sum",
        }))

        # v6-D lambda sweep
        for lam in [0.3, 0.5, 1.0]:
            runs.append((pid, f"v6d_lambda{lam}", {
                "guide_type": "analytic_v6", "esfield_lambda": lam,
                "v6_config": {
                    "sigma_occ": 1.2, "disp_weight": 1.0,
                    "wrong_atom_weight": 0.5, "clash_weight": 0.0,
                    "overfill_weight": 0.0, "min_confidence": 0.3,
                    "top_k": 0, "filter_mixed": False,
                    "cutoff_dist": 5.0,
                },
            }))

        # v6-D wrong-pocket lambda=1.0
        wrong_pid = WRONG_POCKET_MAP[pid]
        runs.append((pid, f"v6d_wrong_{wrong_pid}_lambda1.0", {
            "guide_type": "analytic_v6", "esfield_lambda": 1.0,
            "site_src": "wrong", "wrong_pid": wrong_pid,
            "v6_config": {
                "sigma_occ": 1.2, "disp_weight": 1.0,
                "wrong_atom_weight": 0.5, "clash_weight": 0.0,
                "overfill_weight": 0.0, "min_confidence": 0.3,
                "top_k": 0, "filter_mixed": False,
                "cutoff_dist": 5.0,
            },
        }))

        # v6-D random-matrix lambda=1.0
        runs.append((pid, "v6d_random_matrix_lambda1.0", {
            "guide_type": "analytic_v6", "esfield_lambda": 1.0,
            "site_src": "random",
            "v6_config": {
                "sigma_occ": 1.2, "disp_weight": 1.0,
                "wrong_atom_weight": 0.5, "clash_weight": 0.0,
                "overfill_weight": 0.0, "min_confidence": 0.3,
                "top_k": 0, "filter_mixed": False,
                "cutoff_dist": 5.0,
            },
        }))

    return runs


def run_test_v6(model, potential, progress, dry_run=False):
    """Run all 7 conditions on 5 pockets for v6-D displacement test."""
    all_runs = build_run_list()

    if dry_run:
        print(f"Pockets: {MECHANISM_POCKETS}")
        print(f"Would run {len(all_runs)} conditions "
              f"({len(all_runs) * GEN_PARAMS['num_samples']} generations)")
        for pid, cond, kwargs in all_runs[:14]:
            lam = kwargs.get("esfield_lambda", 0)
            gt = kwargs.get("guide_type", "learned_v5")
            ss = kwargs.get("site_src", "correct")
            print(f"  {pid}: {cond} (λ={lam}, guide={gt}, site={ss})")
        print(f"  ... ({len(all_runs)} total)")
        return

    n_done = 0
    n_total = len(all_runs)

    # Count already-done
    for pid, cond_name, kwargs in all_runs:
        key = f"{pid}_{cond_name}"
        if progress.get(key) == "done":
            n_done += 1
            continue
        sdf_path = OUTPUT_BASE / key / "molecules.sdf"
        if sdf_path.exists():
            progress[key] = "done"
            n_done += 1

    if n_done == n_total:
        print("All runs complete.")
        return

    print(f"\n{'='*60}")
    print(f"v6-D Displacement Test: {n_total - n_done} remaining of {n_total}")
    print(f"{'='*60}")

    for pid, cond_name, kwargs in all_runs:
        key = f"{pid}_{cond_name}"
        if progress.get(key) == "done":
            continue

        paths = get_pocket_paths(pid)

        # Determine site map
        site_src = kwargs.pop("site_src", "correct")
        wrong_pid = kwargs.pop("wrong_pid", None)
        if site_src == "random":
            site_map = str(SITE_MAPS_DIR / "random" / f"{pid}_site_map.json")
        elif site_src == "wrong":
            site_map = str(SITE_MAPS_DIR / "correct" / f"{wrong_pid}_site_map.json")
        else:
            site_map = str(SITE_MAPS_DIR / "correct" / f"{pid}_site_map.json")

        guide_type = kwargs.get("guide_type", "learned_v5")
        esfield_lambda = kwargs.get("esfield_lambda", 1.0)
        v6_config = kwargs.get("v6_config", None)

        # For random-matrix v6-D, we'll post-hoc modify the guide
        # (handled by the analysis script comparing behavior)
        is_random_matrix = "random_matrix" in cond_name

        print(f"\n[{n_done+1}/{n_total}] {pid} {cond_name} (λ={esfield_lambda})")
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=site_map,
                output_dir=str(OUTPUT_BASE / key),
                esfield_lambda=esfield_lambda,
                guide_type=guide_type,
                v6_config=v6_config,
                **GEN_PARAMS,
            )
            print(f"  Valid: {result['valid_count']}/{GEN_PARAMS['num_samples']}, "
                  f"QED: {result['metrics']['qed_mean']:.3f}, "
                  f"{result['elapsed']:.0f}s")
            progress[key] = "done"
            save_progress(progress)
            n_done += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {n_done}/{n_total}")


def main():
    parser = argparse.ArgumentParser(
        description="v6-D HEW Displacement Mechanism Test (5 pockets, 7 conditions)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without running")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--force", action="store_true",
                        help="Ignore saved progress, re-run all")
    args = parser.parse_args()

    GEN_PARAMS["device"] = args.device
    GEN_PARAMS["gen_batch_size"] = args.batch_size

    progress = {} if args.force else load_progress()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no molecules generated")
        print("=" * 60)
        run_test_v6(None, None, {}, dry_run=True)
        return

    print("Loading models...")
    potential = load_esfield_potential(POTENTIAL_DEFAULT_CKPT, device=args.device)
    model = load_drugflow_model(DRUGFLOW_CKPT, device=args.device)
    print(f"DrugFlow: {sum(p.numel() for p in model.parameters()):,} params")

    t0 = time.time()
    try:
        run_test_v6(model, potential, progress)
    finally:
        save_progress(progress)

    print(f"\nTotal wall time: {(time.time() - t0) / 3600:.1f}h")


if __name__ == "__main__":
    main()
