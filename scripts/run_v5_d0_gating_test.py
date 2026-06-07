#!/usr/bin/env python3
"""Quick 2-pocket test: d0 sweep + HEW gating comparison.

Tests on 3ohi (strong responder) and 2clh (weak responder):
  baseline
  v5 d0=3.0 λ=1.0 (original)
  v5 d0=1.5 λ=1.0 (water replacement well)
  v5 d0=1.5 λ=1.0 + nearest-site gating
  v5 d0=1.5 λ=0.5 + nearest-site gating
"""
import json, sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drugflow_esfield_guide import (
    load_esfield_potential, load_drugflow_model, generate_molecules,
    POTENTIAL_DEFAULT_CKPT, DRUGFLOW_CKPT,
)

POCKETS = ["3ohi", "2clh"]
OUTPUT_BASE = ROOT / "experiments/pdbbind_water_sites/v5_d0_gating_test"
TEST_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"
SITE_MAPS = ROOT / "experiments/pdbbind_water_sites/test_sites"

CONDITIONS = [
    ("baseline", 0.0, None, "all"),
    ("d3.0_lam1.0", 1.0, 3.0, "all"),
    ("d1.5_lam1.0", 1.0, 1.5, "all"),
    ("d1.5_lam1.0_nearest", 1.0, 1.5, "nearest"),
    ("d1.5_lam0.5_nearest", 0.5, 1.5, "nearest"),
]

GEN_PARAMS = {
    "num_samples": 20, "timesteps": 40, "gen_batch_size": 5,
    "guidance_start": 0.4, "guidance_end": 0.85, "device": "cuda:0",
}


def get_paths(pid):
    test_pockets = json.loads(TEST_POCKETS_JSON.read_text())
    for p in test_pockets:
        if p["pdb_id"] == pid:
            d = Path(p["dir"])
            return {
                "protein_pdb": str(d / f"{pid}_protein.pdb"),
                "ref_ligand": str(d / f"{pid}_ligand.sdf"),
                "pdb_id": pid,
            }
    raise ValueError(pid)


def main():
    print("Loading models...")
    pot = load_esfield_potential(POTENTIAL_DEFAULT_CKPT, "cuda:0")
    model = load_drugflow_model(DRUGFLOW_CKPT, "cuda:0")

    n_total = len(POCKETS) * len(CONDITIONS)
    n_done = 0

    for pid in POCKETS:
        paths = get_paths(pid)
        site_map = str(SITE_MAPS / "correct" / f"{pid}_site_map.json")
        sm = json.load(open(site_map))
        n_hew = sum(1 for s in sm["sites"] if s["site_type"] == "high_energy_water")

        for cond_name, lam, d0, gating in CONDITIONS:
            n_done += 1
            out_dir = OUTPUT_BASE / f"{pid}_{cond_name}"
            if (out_dir / "molecules.sdf").exists():
                print(f"[{n_done}/{n_total}] {pid} {cond_name}: SKIP (exists)")
                continue

            print(f"[{n_done}/{n_total}] {pid} {cond_name} (λ={lam}, d0={d0}, gate={gating}, n_hew={n_hew})")
            try:
                r = generate_molecules(
                    model, pot,
                    protein_pdb=paths["protein_pdb"],
                    ref_ligand=paths["ref_ligand"],
                    site_map_path=site_map,
                    output_dir=str(out_dir),
                    esfield_lambda=lam,
                    d0_override=d0,
                    hew_gating=gating,
                    **GEN_PARAMS,
                )
                print(f"  Valid: {r['valid_count']}/{GEN_PARAMS['num_samples']}, "
                      f"QED: {r['metrics']['qed_mean']:.3f}, {r['elapsed']:.0f}s")
            except Exception as e:
                print(f"  FAILED: {e}")

    print(f"\nDone. Analyze with:")
    print(f"  PYTHONPATH=src python scripts/analyze_v5_d0_gating.py")


if __name__ == "__main__":
    main()
