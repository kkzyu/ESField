#!/usr/bin/env python3
"""DEPRECATED — use drugflow_esfield_guide.py or run_experiment_matrix.py instead.

This script runs generation sample-by-sample via subprocess, which is ~4x slower
than the batch generation in drugflow_esfield_guide.py. Kept for reference only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--potential-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--volume", type=float, default=300)
    parser.add_argument("--area", type=float, default=300)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--esfield-lambda", type=float, default=0.5)
    parser.add_argument("--pos-grad-w", type=float, default=175.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sub_dirs = []

    script = ROOT / "scripts/run_esfield_guided.py"

    start_time = time.time()
    for sample_i in range(args.num_samples):
        sample_dir = output_dir / f"sample_{sample_i:03d}"
        sample_dir.mkdir(exist_ok=True)
        sub_dirs.append(sample_dir)

        cmd = [
            sys.executable, str(script),
            "--protein-pdb", args.protein_pdb,
            "--site-map", args.site_map,
            "--potential-ckpt", args.potential_ckpt,
            "--output-dir", str(sample_dir),
            "--volume", str(args.volume),
            "--area", str(args.area),
            "--num-samples", "1",
            "--batch-size", "1",
            "--esfield-lambda", str(args.esfield_lambda),
            "--pos-grad-w", str(args.pos_grad_w),
            "--device", args.device,
        ]

        print(f"[{sample_i+1}/{args.num_samples}] Generating sample {sample_i}...", end=" ", flush=True)
        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=str(ROOT),
            timeout=600,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"OK ({elapsed:.1f}s)")
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            stderr = result.stderr[-500:] if result.stderr else ""
            print(f"  STDERR: {stderr}")

    total_time = time.time() - start_time
    print(f"\nAll {args.num_samples} samples complete in {total_time:.1f}s")

    # Combine results
    import torch
    all_pos, all_v = [], []
    n_success = 0
    for d in sub_dirs:
        result_pt = d / "result.pt"
        if result_pt.exists():
            r = torch.load(str(result_pt), map_location="cpu")
            all_pos.append(r["pos"])
            all_v.append(r["v"])
            n_success += 1
            # Copy SDF
            for sdf_f in (d / "sdf").glob("*.sdf"):
                import shutil
                shutil.copy(str(sdf_f), str(output_dir / "sdf" / sdf_f.name))
        else:
            print(f"  WARNING: no result.pt in {d}")

    if all_pos:
        (output_dir / "sdf").mkdir(exist_ok=True)
        combined = {"pos": torch.cat(all_pos, dim=0), "v": torch.cat(all_v, dim=0)}
        torch.save(combined, output_dir / "result.pt")
        print(f"Combined {n_success}/{args.num_samples} results -> {output_dir}/result.pt")
    else:
        print("No successful results to combine")


if __name__ == "__main__":
    main()
