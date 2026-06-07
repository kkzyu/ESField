#!/usr/bin/env python3
"""Batch run TargetDiff baseline vs ESField guided on multiple pockets.

Each sample runs in a subprocess to avoid GPU memory accumulation.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TD_ROOT = Path("/root/baselines/TargetDiff/code/targetdiff-main")

POCKETS = [
    "TIAM1_HUMAN_840_931_0",
    "FKB1A_HUMAN_2_108_0",
    "UBE2T_HUMAN_1_156_0",
]


def run_td_baseline(pocket_id, rec_pdb, output_dir, n_samples):
    """Run TargetDiff baseline via its sample_for_pocket.py."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    cfg_path = TD_ROOT / "configs/sampling.yml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["sample"]["num_samples"] = n_samples
    cfg["sample"]["batch_size"] = 1
    tmp_cfg = output_dir / "sampling.yml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)

    cmd = [
        sys.executable, str(TD_ROOT / "scripts/sample_for_pocket.py"),
        str(tmp_cfg),
        "--pdb_path", str(rec_pdb),
        "--batch_size", "1",
        "--result_path", str(output_dir),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(TD_ROOT), timeout=600,
                            env={**__import__("os").environ, "PYTHONPATH": str(TD_ROOT)})
    elapsed = time.time() - t0
    ok = result.returncode == 0
    return {"ok": ok, "elapsed": elapsed, "stderr": result.stderr[-200:] if result.stderr else ""}


def run_td_guided(pocket_id, rec_pdb, site_map, output_dir, n_samples, esfield_lambda):
    """Run TargetDiff + ESField via targetdiff_esfield_guide.py (one subprocess per sample)."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    pot_ckpt = str(ROOT / "experiments/potential_training/train_gpu/compatibility_potential_epoch_0200.pt")
    script = str(ROOT / "scripts/targetdiff_esfield_guide.py")

    all_ok = True
    total_elapsed = 0
    for si in range(n_samples):
        sample_dir = output_dir / f"sample_{si:03d}"
        sample_dir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, script,
            "--protein-pdb", str(rec_pdb),
            "--site-map", str(site_map),
            "--potential-ckpt", pot_ckpt,
            "--output-dir", str(sample_dir),
            "--num-samples", "1",
            "--esfield-lambda", str(esfield_lambda),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = result.returncode == 0
        if not ok:
            all_ok = False
            print(f"  Sample {si} FAILED: {result.stderr[-200:]}")
        total_elapsed += (time.time() - time.time())  # approximate
    return {"ok": all_ok, "elapsed": total_elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--esfield-lambda", type=float, default=0.5)
    parser.add_argument("--pockets", nargs="*")
    args = parser.parse_args()

    pockets_to_run = args.pockets or POCKETS
    test_set = Path("/root/autodl-tmp/data/test_sets/CrossDocked_test_set")
    site_maps = Path("/root/ESField/experiments/potential_training/site_maps")

    results = {}
    for pid in pockets_to_run:
        rec = test_set / pid
        rec_pdbs = sorted(rec.glob("*_rec.pdb")) if rec.exists() else []
        if not rec_pdbs:
            print(f"{pid}: no receptor PDB, skipping"); continue
        rec_pdb = rec_pdbs[0]
        sm = site_maps / f"{pid}_site_map.json"
        if not sm.exists():
            print(f"{pid}: no site map, skipping"); continue

        exp_dir = ROOT / f"experiments/td_comparison/{pid}"
        print(f"\n{'='*50}\n{pid}")

        # Baseline
        print("  Baseline...", end=" ", flush=True)
        b_r = run_td_baseline(pid, rec_pdb, exp_dir / "baseline", args.n_samples)
        print(f"{'OK' if b_r['ok'] else 'FAIL'} ({b_r['elapsed']:.0f}s)")

        # Guided
        print("  Guided...", end=" ", flush=True)
        g_r = run_td_guided(pid, rec_pdb, sm, exp_dir / "guided", args.n_samples, args.esfield_lambda)
        print(f"{'OK' if g_r['ok'] else 'FAIL'}")

        results[pid] = {"baseline_ok": b_r["ok"], "guided_ok": g_r["ok"]}

    print(f"\n{'='*50}\nSUMMARY")
    for pid, r in results.items():
        print(f"  {pid[:40]}: BL={'OK' if r['baseline_ok'] else 'FAIL'}, GD={'OK' if r['guided_ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
