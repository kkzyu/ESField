#!/usr/bin/env python3
"""sum_norm guidance 5-pocket verification experiment.

Tests whether sum_norm (per-atom normalized energy) aggregation improves
molecule-level ranking quality over simple sum aggregation in generation.

Experiment matrix (5 pockets × 7 conditions = 700 gens):

  | 条件 | λ | Aggregation | Site |
  |------|---|-------------|------|
  | baseline | 0 | — | correct |
  | v5-sum | 1.0 | sum | correct |
  | v5-sum_norm | 1.0 | sum_norm | correct |
  | v5-sum_norm | 5.0 | sum_norm | correct |
  | v5-sum_norm | 10.0 | sum_norm | correct |
  | v5-sum_norm | 20.0 | sum_norm | correct |
  | v5-sum_norm wrong | 10.0 | sum_norm | wrong-pocket |

Go/No-Go criteria (≥3/5):
  - 3mfw or 1sle at least one improves from flat
  - sum_norm mean |d-3.0| better than old sum
  - sum_norm top-10% |d-3.0| better than old sum
  - wrong-pocket doesn't produce equal improvement
  - QED drop < 0.03

Usage:
  python scripts/run_sum_norm_experiment.py --dry-run
  python scripts/run_sum_norm_experiment.py
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drugflow_esfield_guide import (
    load_esfield_potential, load_drugflow_model, generate_molecules,
    POTENTIAL_DEFAULT_CKPT, DRUGFLOW_CKPT,
)

POCKETS = ["3ohi", "2clh", "3mfw", "4bis", "1sle"]

# Experiment conditions: (name, lambda, aggregation, site_type)
CONDITIONS = [
    ("baseline", 0.0, "sum", "correct"),               # baseline
    ("v5-sum_lambda1.0", 1.0, "sum", "correct"),        # old control
    ("v5-sumnorm_lambda1.0", 1.0, "sum_norm", "correct"), # direct norm
    ("v5-sumnorm_lambda5.0", 5.0, "sum_norm", "correct"), # compensate
    ("v5-sumnorm_lambda10.0", 10.0, "sum_norm", "correct"),# near sum scale
    ("v5-sumnorm_lambda20.0", 20.0, "sum_norm", "correct"),# strong
    ("v5-sumnorm_wrong_lambda10.0", 10.0, "sum_norm", "wrong"), # control
]

WRONG_POCKET_MAP = {"3ohi": "5g60", "2clh": "1sle", "3mfw": "2jkr", "4bis": "3r01", "1sle": "2wgi"}

GEN_PARAMS = {
    "num_samples": 20,
    "timesteps": 40,
    "gen_batch_size": 5,
    "guidance_start": 0.4,
    "guidance_end": 0.85,
    "device": "cuda:0",
}

OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/v5_sum_norm_test"
TEST_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"
SITE_MAPS_DIR = ROOT / "experiments/pdbbind_water_sites/test_sites"


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
            }
    raise ValueError(f"Pocket {pdb_id} not found")


def run_experiment(model, potential, progress, dry_run=False):
    all_runs = []
    for pid in POCKETS:
        for cond_name, lam, agg, site_type in CONDITIONS:
            wrong_pid = WRONG_POCKET_MAP[pid] if site_type == "wrong" else None
            all_runs.append((pid, cond_name, lam, agg, site_type, wrong_pid))

    if dry_run:
        n_effective = sum(1 for _, c, _, _, _, _ in all_runs if "baseline" in c)
        n_full = sum(1 for _, c, _, _, _, _ in all_runs if "baseline" not in c)
        print(f"Would run {len(all_runs)} conditions ({len(all_runs) * GEN_PARAMS['num_samples']} generations):")
        print(f"  {n_effective} baseline runs (shared) + {n_full} guided runs")
        for pid, cond_name, lam, agg, site_type, wp in all_runs:
            print(f"  {pid}: {cond_name} λ={lam} agg={agg} site={site_type}")
        return

    # Count already-done runs
    n_done = 0
    n_total = len(all_runs)
    for pid, cond_name, lam, agg, site_type, wrong_pid in all_runs:
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
    print(f"sum_norm Experiment: {n_total - n_done} remaining of {n_total}")
    print(f"{'='*60}")

    for pid, cond_name, lam, agg, site_type, wrong_pid in all_runs:
        key = f"{pid}_{cond_name}"
        if progress.get(key) == "done":
            continue

        paths = get_pocket_paths(pid)

        # Resolve site map
        if site_type == "wrong":
            site_map = str(SITE_MAPS_DIR / "correct" / f"{wrong_pid}_site_map.json")
        else:
            site_map = str(SITE_MAPS_DIR / "correct" / f"{pid}_site_map.json")

        print(f"\n[{n_done+1}/{n_total}] {pid} {cond_name} (λ={lam}, agg={agg})")
        try:
            result = generate_molecules(
                model, potential,
                protein_pdb=paths["protein_pdb"],
                ref_ligand=paths["ref_ligand"],
                site_map_path=site_map,
                output_dir=str(OUTPUT_BASE / key),
                esfield_lambda=lam,
                aggregation=agg,
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
        print(f"Pockets: {POCKETS}")
        print(f"Conditions: {len(CONDITIONS)} ({len(POCKETS) * len(CONDITIONS)} total runs)")
        run_experiment(None, None, {}, dry_run=True)
        return

    print("Loading models...")
    potential = load_esfield_potential(POTENTIAL_DEFAULT_CKPT, device=args.device)
    model = load_drugflow_model(DRUGFLOW_CKPT, device=args.device)

    t0 = time.time()
    try:
        run_experiment(model, potential, progress)
    finally:
        save_progress(progress)

    print(f"\nTotal wall time: {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
