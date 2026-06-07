#!/usr/bin/env python3
"""5-pocket HEW-focused mechanism validation for Potential v5.

Tests whether v5 guidance actually reduces HEW-compatible nearest atom distance
in real DrugFlow generation. Primary metric: Δd_HEW (not POSU).

Conditions (7 total per pocket):
  baseline (λ=0)
  v5-correct  λ=0.1, 0.3, 0.5, 1.0
  v5-random   λ=0.5
  v5-wrong-pocket λ=0.5

Pockets: 3 strong HEW signal + 1 medium + 1 hard = 5 pockets
Total: 5 pockets × 7 conditions × 20 samples = 700 generations (~0.9h GPU)

Usage:
  python scripts/run_v5_mechanism_test.py --dry-run    # Show plan
  python scripts/run_v5_mechanism_test.py              # Run all
"""

from __future__ import annotations
import argparse, json, os, sys, time, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drugflow_esfield_guide import (
    load_esfield_potential, load_drugflow_model, generate_molecules,
    POTENTIAL_DEFAULT_CKPT, DRUGFLOW_CKPT,
)

# Selected 5 pockets (3 strong HEW + 1 medium + 1 hard)
# Based on Step 1 + native ligand POSU diagnosis:
#   Strong HEW signal: 3ohi, 2clh, 3mfw (HEWU correct>>random)
#   Medium: 4bis
#   Hard: 1sle
MECHANISM_POCKETS = ["3ohi", "2clh", "3mfw", "4bis", "1sle"]

LAMBDA_SWEEP = [0.1, 0.3, 0.5, 1.0]

GEN_PARAMS = {
    "num_samples": 20,
    "timesteps": 40,
    "gen_batch_size": 5,
    "guidance_start": 0.4,
    "guidance_end": 0.85,
    "device": "cuda:0",
}

OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/v5_mechanism_test"
TEST_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"
SITE_MAPS_DIR = ROOT / "experiments/pdbbind_water_sites/test_sites"

# Wrong-pocket mapping (deterministic)
WRONG_POCKET_MAP = {"3ohi": "5g60", "2clh": "1sle", "3mfw": "2jkr", "4bis": "3r01", "1sle": "2wgi"}


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


def run_test(model, potential, progress, dry_run=False):
    """Run all 7 conditions on 5 pockets."""
    all_runs = []

    for pid in MECHANISM_POCKETS:
        # Baseline (λ=0)
        all_runs.append((pid, "baseline", 0.0, "correct", None))

        # v5-correct λ sweep
        for lam in LAMBDA_SWEEP:
            all_runs.append((pid, f"v5_lambda{lam}", lam, "correct", None))

        # v5-random λ=0.5
        all_runs.append((pid, "v5_random_lambda0.5", 0.5, "random", None))

        # v5-wrong-pocket λ=0.5
        wrong_pid = WRONG_POCKET_MAP[pid]
        all_runs.append((pid, f"v5_wrong_{wrong_pid}_lambda0.5", 0.5, "wrong", wrong_pid))

    if dry_run:
        print(f"Would run {len(all_runs)} conditions ({len(all_runs) * GEN_PARAMS['num_samples']} generations)")
        for pid, cond, lam, site_src, wp in all_runs[:15]:
            print(f"  {pid}: {cond}")
        print(f"  ... ({len(all_runs)} total)")
        return

    n_done = 0
    n_total = len(all_runs)
    for pid, cond_name, lam, site_src, wrong_pid in all_runs:
        key = f"{pid}_{cond_name}"
        if progress.get(key) == "done":
            n_done += 1
            continue

        sdf_path = OUTPUT_BASE / key / "molecules.sdf"
        if sdf_path.exists():
            progress[key] = "done"
            n_done += 1
            continue

    if n_done == n_total:
        print("All runs complete.")
        return

    print(f"\n{'='*60}")
    print(f"V5 Mechanism Test: {n_total - n_done} remaining of {n_total}")
    print(f"{'='*60}")

    for pid, cond_name, lam, site_src, wrong_pid in all_runs:
        key = f"{pid}_{cond_name}"
        if progress.get(key) == "done":
            continue

        paths = get_pocket_paths(pid)

        # Determine site map path
        if site_src == "correct":
            site_map = str(SITE_MAPS_DIR / "correct" / f"{pid}_site_map.json")
        elif site_src == "random":
            site_map = str(SITE_MAPS_DIR / "random" / f"{pid}_site_map.json")
        else:  # wrong
            site_map = str(SITE_MAPS_DIR / "correct" / f"{wrong_pid}_site_map.json")

        print(f"\n[{n_done+1}/{n_total}] {pid} {cond_name} (λ={lam})")
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=site_map,
                output_dir=str(OUTPUT_BASE / key),
                esfield_lambda=lam,
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
            import traceback; traceback.print_exc()

    print(f"\nDone: {n_done}/{n_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    GEN_PARAMS["device"] = args.device
    GEN_PARAMS["gen_batch_size"] = args.batch_size

    progress = {} if args.force else load_progress()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no molecules generated")
        print("=" * 60)
        print(f"Pockets: {MECHANISM_POCKETS}")
        print(f"Conditions: baseline + v5 λ={LAMBDA_SWEEP} + random + wrong-pocket")
        print(f"Total: {len(MECHANISM_POCKETS)} × {len(LAMBDA_SWEEP)+3} × {GEN_PARAMS['num_samples']}")
        run_test(None, None, {}, dry_run=True)
        return

    print("Loading models...")
    potential = load_esfield_potential(POTENTIAL_DEFAULT_CKPT, device=args.device)
    model = load_drugflow_model(DRUGFLOW_CKPT, device=args.device)

    t0 = time.time()
    try:
        run_test(model, potential, progress)
    finally:
        save_progress(progress)

    print(f"\nTotal wall time: {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
