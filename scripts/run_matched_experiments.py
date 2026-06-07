#!/usr/bin/env python3
"""Run matched baseline (PAFlow only) and guided (PAFlow + ESField) experiments.

For each pocket, runs N baseline samples and N guided samples with matched settings.
Results are saved for later comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAFLOW_ROOT = Path("/root/PAFlow-main")

# Selected pockets: diverse sizes, all have site maps
POCKETS = {
    "ABL2_HUMAN_274_551_0": {
        "rec": PAFLOW_ROOT / "data/test_set/ABL2_HUMAN_274_551_0/4xli_B_rec.pdb",
        "site_map": ROOT / "experiments/potential_training/site_maps/ABL2_HUMAN_274_551_0_site_map.json",
    },
    "FKB1A_HUMAN_2_108_0": {
        "rec": PAFLOW_ROOT / "data/test_set/FKB1A_HUMAN_2_108_0/1d7j_A_rec.pdb",
        "site_map": ROOT / "experiments/potential_training/site_maps/FKB1A_HUMAN_2_108_0_site_map.json",
    },
    "TNKS2_HUMAN_948_1162_0": {
        "rec": PAFLOW_ROOT / "data/test_set/TNKS2_HUMAN_948_1162_0/5aeh_A_rec.pdb",
        "site_map": ROOT / "experiments/potential_training/site_maps/TNKS2_HUMAN_948_1162_0_site_map.json",
    },
    "TIAM1_HUMAN_840_931_0": {
        "rec": PAFLOW_ROOT / "data/test_set/TIAM1_HUMAN_840_931_0/4gvd_A_rec.pdb",
        "site_map": ROOT / "experiments/potential_training/site_maps/TIAM1_HUMAN_840_931_0_site_map.json",
    },
}


def run_baseline(protein_id, rec_pdb, output_dir, n_samples, volume, area, device):
    """Run PAFlow baseline sampling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PAFLOW_ROOT / "scripts/sample_for_pocket.py"),
        "--config", str(PAFLOW_ROOT / "configs/sampling_guide.yml"),
        "--pdb_path", str(rec_pdb),
        "--volume", str(volume),
        "--area", str(area),
        "--device", device,
        "--batch_size", "1",
        "--result_path", str(output_dir),
    ]

    # Patch num_samples in config
    import yaml
    cfg_path = PAFLOW_ROOT / "configs/sampling_guide.yml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["sample"]["num_samples"] = n_samples
    tmp_cfg = output_dir / "sampling_config.yml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)
    cmd[2] = str(tmp_cfg)

    print(f"  Baseline: {cmd[0]} ...")
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(PAFLOW_ROOT), timeout=600,
        env={**__import__('os').environ, "PYTHONPATH": str(PAFLOW_ROOT) + ":" + str(PAFLOW_ROOT / "scripts")},
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        # Count complete molecules
        n_complete = 0
        sdf_dir = output_dir / "sdf"
        if sdf_dir.exists():
            n_complete = len(list(sdf_dir.glob("*.sdf")))
        print(f"  Baseline OK ({elapsed:.1f}s, {n_complete} complete)")
        return True, n_complete
    else:
        print(f"  Baseline FAILED: {result.stderr[-200:]}")
        return False, 0


def run_guided(protein_id, rec_pdb, site_map, output_dir, n_samples, volume, area, device, esfield_lambda, pos_grad_w):
    """Run ESField guided generation via batch subprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)
    guided_script = ROOT / "scripts/run_guided_batch.py"
    pot_ckpt = ROOT / "experiments/potential_training/train_gpu/compatibility_potential_epoch_0200.pt"

    cmd = [
        sys.executable, str(guided_script),
        "--protein-pdb", str(rec_pdb),
        "--site-map", str(site_map),
        "--potential-ckpt", str(pot_ckpt),
        "--output-dir", str(output_dir),
        "--volume", str(volume),
        "--area", str(area),
        "--num-samples", str(n_samples),
        "--esfield-lambda", str(esfield_lambda),
        "--pos-grad-w", str(pos_grad_w),
        "--device", device,
    ]

    print(f"  Guided: esfield_lambda={esfield_lambda}, pos_grad_w={pos_grad_w}")
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(ROOT), timeout=1200,
        env={**__import__('os').environ, "PYTHONPATH": str(ROOT / "src")},
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        stdout_tail = result.stdout.strip().split("\n")[-5:]
        print(f"  Guided OK ({elapsed:.1f}s)")
        for line in stdout_tail:
            if line.strip():
                print(f"    {line.strip()}")
        return True
    else:
        print(f"  Guided FAILED ({elapsed:.1f}s)")
        for line in result.stderr.strip().split("\n")[-3:]:
            print(f"    {line.strip()}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--volume", type=float, default=300)
    parser.add_argument("--area", type=float, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esfield-lambda", type=float, default=0.5)
    parser.add_argument("--pos-grad-w", type=float, default=350.0)
    parser.add_argument("--pockets", nargs="*", default=None,
                       help="Specific pockets to run (default: all)")
    args = parser.parse_args()

    exp_dir = ROOT / "experiments/matched_experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    pockets_to_run = args.pockets if args.pockets else list(POCKETS.keys())
    results = {}

    for pid in pockets_to_run:
        if pid not in POCKETS:
            print(f"Unknown pocket: {pid}")
            continue

        info = POCKETS[pid]
        print(f"\n{'='*60}")
        print(f"Pocket: {pid}")

        # Check site map
        if not info["site_map"].exists():
            print(f"  SKIP: site map not found at {info['site_map']}")
            continue

        with open(info["site_map"]) as f:
            sm = json.load(f)
        n_sites = sm.get("n_sites", len(sm.get("sites", [])))
        n_atoms = 0
        with open(info["rec"]) as f:
            n_atoms = sum(1 for l in f if l.startswith("ATOM") or l.startswith("HETATM"))
        print(f"  {n_atoms} atoms, {n_sites} sites")

        # Run baseline
        baseline_dir = exp_dir / pid / "baseline"
        b_ok, b_complete = run_baseline(
            pid, info["rec"], baseline_dir, args.n_samples,
            args.volume, args.area, args.device)

        # Run guided
        guided_dir = exp_dir / pid / "guided"
        g_ok = run_guided(
            pid, info["rec"], info["site_map"], guided_dir, args.n_samples,
            args.volume, args.area, args.device,
            args.esfield_lambda, args.pos_grad_w)

        results[pid] = {
            "n_atoms": n_atoms,
            "n_sites": n_sites,
            "baseline_ok": b_ok,
            "baseline_complete": b_complete,
            "guided_ok": g_ok,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'Pocket':<30} {'Atoms':>6} {'Sites':>6} {'BL OK':>6} {'BL Cpl':>6} {'GD OK':>6}")
    print("-" * 70)
    for pid, r in results.items():
        name = pid[:28]
        print(f"{name:<30} {r['n_atoms']:>6} {r['n_sites']:>6} "
              f"{str(r['baseline_ok']):>6} {r['baseline_complete']:>6} "
              f"{str(r['guided_ok']):>6}")

    # Save summary
    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
