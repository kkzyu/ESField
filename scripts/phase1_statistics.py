#!/usr/bin/env python3
"""Phase 1 Statistics Module — evaluate anchor generation capability.

Runs Phase 1 independently N times per pocket (default 100), recording:
  - Success rate (fraction of runs that produce ≥1 anchor)
  - Average number of anchors per successful run
  - Per-HEW-site coverage (which sites are covered across runs)
  - E_site values
  - Anchor atom type distribution

Output: phase1_stats.csv + phase1_stats.json

Usage:
    # From DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/phase1_statistics.py \
        --protein-pdb /path/to/pocket.pdb \
        --site-map /path/to/site_map.json \
        --ref-ligand /path/to/ligand.sdf \
        --output-dir results/phase1_stats/3mfw \
        --n-runs 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Setup DrugFlow paths ──
DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"
sys.path.insert(0, os.path.join(DRUGFLOW_DIR, "src"))
sys.path.insert(0, DRUGFLOW_DIR)

from src.model import lightning as lmod
from src.data.data_utils import process_raw_pair, TensorDict
from src.data.dataset import ProcessedLigandPocketDataset
from torch.utils.data import DataLoader
from functools import partial
from rdkit import Chem
from Bio.PDB import PDBParser

# ── ESField imports ──
ESFIELD_ROOT = "/root/ESField"
sys.path.insert(0, os.path.join(ESFIELD_ROOT, "src"))
sys.path.insert(0, ESFIELD_ROOT)

from guidance.latent_guidance import (
    SiteCompatibilityEnergy, build_site_energy_from_map,
    classify_hew_environment, ATOM_TYPE_VOCAB, ATOM_TYPE_TO_IDX,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    Phase1Config, _Phase1GuideFn, _compute_diagnostics,
    _extract_anchors, _tensors_from_rdmol,
)

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"


def load_model(ckpt_path, device="cuda:0"):
    warnings.filterwarnings("ignore")
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        model = lmod.DrugFlow.load_from_checkpoint(ckpt_path, map_location=device)
    finally:
        torch.load = _orig_load
    return model.to(device).eval()


def process_protein(protein_pdb, ref_ligand, model):
    """Process protein + ligand for DrugFlow."""
    pdb_model = PDBParser(QUIET=True).get_structure("", protein_pdb)[0]
    rdmol = Chem.SDMolSupplier(ref_ligand)[0]
    ref_size = rdmol.GetNumAtoms()

    ligand_raw, pocket_raw = process_raw_pair(
        pdb_model, rdmol, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True)
    ligand_raw["name"] = "ligand"

    collate = partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None)
    data = next(iter(DataLoader(
        [{"ligand": ligand_raw, "pocket": pocket_raw}],
        batch_size=1, collate_fn=collate)))
    return data, ref_size


def run_phase1_statistics(
    model,
    protein_data: dict,
    site_map: dict,
    *,
    n_runs: int = 100,
    n_atoms: int = 4,
    lambda_guide: float = 5.0,
    success_distance: float = 2.5,
    min_compatibility: float = 0.3,
    timesteps: int = 50,
    device: str = "cuda:0",
    verbose: bool = False,
) -> dict:
    """Run Phase 1 N times and collect statistics.

    Each run independently generates a 4-atom fragment and checks for anchors.
    """
    # Build site energy
    site_energy = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(device)

    n_hew_sites = site_energy.n_sites
    if n_hew_sites == 0:
        return {"error": "No HEW sites found", "n_hew_sites": 0}

    # KTS scheduler
    kts = KTSScheduler(alpha0=0.01, beta0=0.01)

    # Phase 1 config
    cfg = Phase1Config(
        n_init_atoms=n_atoms,
        lambda_early=lambda_guide,
        success_distance=success_distance,
        min_compatibility=min_compatibility,
        guidance_start=0.05,
        guidance_end=0.95,
        grad_clip=0.5,
        sigma_distance=3.0,
        anchor_selection="best_per_site",
    )

    guide_fn = _Phase1GuideFn(
        site_energy=site_energy,
        lambda_guide=lambda_guide,
        guidance_start=cfg.guidance_start,
        guidance_end=cfg.guidance_end,
        grad_clip=cfg.grad_clip,
        kts=kts,
    ).to(device)

    results = []
    per_site_coverage = Counter()  # site_idx → count of runs where covered
    all_e_site_values = []
    all_anchor_counts = []
    all_anchor_types = Counter()

    for run_idx in range(n_runs):
        run_start = time.time()

        # Generate one fragment
        with torch.no_grad():
            rdmols, trajectories, _ = model.sample(
                protein_data,
                n_samples=1,
                timesteps=timesteps,
                num_nodes=n_atoms,
                guide_log_prob=guide_fn,
            )

        elapsed = time.time() - run_start
        mol = rdmols[0] if rdmols else None

        if mol is None:
            results.append({
                "run": run_idx,
                "success": False,
                "error": "No molecule generated",
                "elapsed": elapsed,
            })
            continue

        # Extract tensors
        x, h = _tensors_from_rdmol(mol, device=device)
        if x is None:
            results.append({
                "run": run_idx,
                "success": False,
                "error": "Failed to extract tensors",
                "elapsed": elapsed,
            })
            continue

        # Compute diagnostics
        diag = _compute_diagnostics(
            x, h, site_energy,
            success_distance, min_compatibility,
        )

        # Compute E_site for the fragment
        e_site_val = float(site_energy(x, atom_type_probs=F.softmax(h, dim=-1)).cpu())

        run_result = {
            "run": run_idx,
            "success": diag["success"],
            "n_occupied_sites": diag["n_occupied_sites"],
            "best_distance": diag["best_distance"],
            "best_compat": diag["best_compat"],
            "n_hew_sites": n_hew_sites,
            "E_site": e_site_val,
            "elapsed": elapsed,
        }

        if diag["success"]:
            anchors = _extract_anchors(x, h, site_energy, diag, cfg)
            if anchors is not None:
                run_result["n_anchors"] = anchors.n_anchors
                run_result["anchor_types"] = [
                    ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
                    for i in range(anchors.n_anchors)
                ]
                run_result["anchor_distances"] = anchors.distances
                run_result["anchor_compat_scores"] = anchors.compat_scores
                run_result["occupied_sites"] = anchors.occupied_sites

                all_anchor_counts.append(anchors.n_anchors)
                for site_idx in anchors.occupied_sites:
                    per_site_coverage[site_idx] += 1
                for t in run_result["anchor_types"]:
                    all_anchor_types[t] += 1
            else:
                run_result["n_anchors"] = 0
                run_result["anchor_types"] = []
        else:
            run_result["n_anchors"] = 0
            run_result["anchor_types"] = []

        all_e_site_values.append(e_site_val)
        results.append(run_result)

        if verbose and (run_idx + 1) % 10 == 0:
            success_count = sum(1 for r in results if r["success"])
            print(f"  Run {run_idx+1}/{n_runs}: {success_count} successes so far "
                  f"({success_count/(run_idx+1):.1%})")

    # ── Aggregate statistics ──
    success_count = sum(1 for r in results if r["success"])
    success_rate = success_count / n_runs if n_runs > 0 else 0.0

    summary = {
        "n_runs": n_runs,
        "n_hew_sites": n_hew_sites,
        "success_count": success_count,
        "success_rate": success_rate,
        "n_atoms_per_fragment": n_atoms,
        "lambda_guide": lambda_guide,
        "success_distance": success_distance,
        "min_compatibility": min_compatibility,
    }

    if all_anchor_counts:
        summary["mean_anchors_per_success"] = float(np.mean(all_anchor_counts))
        summary["std_anchors_per_success"] = float(np.std(all_anchor_counts))
        summary["median_anchors_per_success"] = float(np.median(all_anchor_counts))
    else:
        summary["mean_anchors_per_success"] = 0.0
        summary["std_anchors_per_success"] = 0.0
        summary["median_anchors_per_success"] = 0.0

    if all_e_site_values:
        summary["mean_E_site"] = float(np.mean(all_e_site_values))
        summary["std_E_site"] = float(np.std(all_e_site_values))

    # Per-site coverage
    per_site_info = {}
    for site_idx in range(n_hew_sites):
        per_site_info[str(site_idx)] = {
            "covered_count": per_site_coverage.get(site_idx, 0),
            "coverage_rate": per_site_coverage.get(site_idx, 0) / success_count
            if success_count > 0 else 0.0,
        }

    # Anchor type distribution
    total_anchors = sum(all_anchor_types.values())
    type_distribution = {
        t: {"count": c, "fraction": c / total_anchors if total_anchors > 0 else 0.0}
        for t, c in all_anchor_types.most_common()
    }

    return {
        "summary": summary,
        "per_site_coverage": per_site_info,
        "anchor_type_distribution": type_distribution,
        "per_run": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Statistics")
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--ref-ligand", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-atoms", type=int, default=4)
    parser.add_argument("--lambda-guide", type=float, default=5.0)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1 Statistics — Anchor Generation Assessment")
    print("=" * 60)
    print(f"  Protein: {args.protein_pdb}")
    print(f"  Site map: {args.site_map}")
    print(f"  N runs: {args.n_runs}")
    print(f"  λ_guide: {args.lambda_guide}")
    print(f"  N atoms: {args.n_atoms}")

    # Load site map
    site_map = json.loads(Path(args.site_map).read_text())
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    print(f"  HEW sites: {len(hew_sites)}")
    for s in hew_sites[:5]:
        env = classify_hew_environment(s)
        print(f"    site {s.get('site_id', '?'):>4}  env={env:<20}  "
              f"conf={s.get('confidence', 1.0):.2f}")

    # Load model
    print(f"\nLoading DrugFlow from {DRUGFLOW_CKPT}...")
    model = load_model(DRUGFLOW_CKPT, device=args.device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Process protein
    print(f"\nProcessing protein {args.protein_pdb}...")
    data, ref_size = process_protein(args.protein_pdb, args.ref_ligand, model)
    print(f"  Reference ligand: {ref_size} atoms")

    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(args.device),
        "pocket": TensorDict(**data["pocket"]).to(args.device),
    }

    # Run statistics
    print(f"\nRunning {args.n_runs} Phase 1 attempts...")
    t0 = time.time()
    stats = run_phase1_statistics(
        model, protein_data, site_map,
        n_runs=args.n_runs,
        n_atoms=args.n_atoms,
        lambda_guide=args.lambda_guide,
        timesteps=args.timesteps,
        device=args.device,
        verbose=args.verbose,
    )
    total_time = time.time() - t0

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        sys.exit(1)

    # ── Save outputs ──
    summary = stats["summary"]
    summary["total_time_s"] = total_time

    # JSON
    json_path = output_dir / "phase1_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  ✓ JSON → {json_path}")

    # CSV
    csv_path = output_dir / "phase1_stats.csv"
    with open(csv_path, "w") as f:
        f.write("run,success,n_occupied_sites,best_distance,best_compat,"
                "n_anchors,E_site,elapsed\n")
        for r in stats["per_run"]:
            f.write(f"{r['run']},{r['success']},{r.get('n_occupied_sites',0)},"
                    f"{r.get('best_distance','')},{r.get('best_compat','')},"
                    f"{r.get('n_anchors',0)},{r.get('E_site','')},{r.get('elapsed','')}\n")
    print(f"  ✓ CSV → {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Success rate:     {summary['success_rate']:.1%} "
          f"({summary['success_count']}/{summary['n_runs']})")
    print(f"  Mean anchors:     {summary['mean_anchors_per_success']:.2f} "
          f"± {summary['std_anchors_per_success']:.2f}")
    print(f"  Mean E_site:      {summary['mean_E_site']:.4f} "
          f"± {summary['std_E_site']:.4f}")
    print(f"  Total time:       {total_time:.1f}s "
          f"({total_time/summary['n_runs']:.1f}s/run)")

    print(f"\n  Per-site coverage:")
    for site_idx, info in stats["per_site_coverage"].items():
        print(f"    site {site_idx}: {info['coverage_rate']:.1%} "
              f"({info['covered_count']}/{summary['n_runs']})")

    print(f"\n  Anchor type distribution:")
    for atype, info in stats["anchor_type_distribution"].items():
        print(f"    {atype:<15} {info['count']:>4} ({info['fraction']:.1%})")


if __name__ == "__main__":
    main()
