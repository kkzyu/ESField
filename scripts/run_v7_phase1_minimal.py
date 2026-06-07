#!/usr/bin/env python3
"""Minimal v7 Phase 1 GPU test — smallest memory footprint.

Tests whether DrugFlow can generate a 4-atom fragment that occupies
a candidate HEW site on the 2g63 pocket.

Usage:
    PYTHONPATH=src python scripts/run_v7_phase1_minimal.py
"""

import json, os, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # For 'import scripts'

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

# v7 imports
from guidance.latent_guidance import (
    SiteCompatibilityEnergy, build_site_energy_from_map,
    classify_hew_environment, ATOM_TYPE_VOCAB, ATOM_TYPE_TO_IDX,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    Phase1Config, _Phase1GuideFn, _compute_diagnostics, _extract_anchors,
)

# DrugFlow integration
from scripts.drugflow_esfield_guide import (
    load_drugflow_model, process_protein_for_drugflow,
)

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
SITE_MAP = f"{ROOT}/experiments/pdbbind_water_sites/site_maps/2g63_site_map.json"
PROTEIN_PDB = "/root/autodl-tmp/data/PDB/P-L/2001-2010/2g63/2g63_protein.pdb"
REF_LIGAND = "/root/autodl-tmp/data/PDB/P-L/2001-2010/2g63/2g63_ligand.sdf"
DEVICE = "cuda:0"

def main():
    print("=" * 60)
    print("v7 Phase 1 — Minimal GPU Test")
    print("=" * 60)

    # 1. Load site map
    print("\n[1] Loading site map...")
    site_map = json.load(open(SITE_MAP))
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    for s in hew_sites:
        env = classify_hew_environment(s)
        print(f"  Site {s['site_id']}: {env}, conf={s['confidence']:.2f}")
    print(f"  Total: {len(hew_sites)} HEW sites")

    # 2. Build site energy (only top-1 HEW for minimal memory)
    print("\n[2] Building SiteCompatibilityEnergy (top-1 HEW)...")
    energy_fn = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
        top_k=1,  # Only 1 site for minimal GPU
    ).to(DEVICE)
    print(f"  Registered {energy_fn.n_sites} HEW site(s)")

    # 3. Load DrugFlow
    print("\n[3] Loading DrugFlow model...")
    t0 = time.time()
    model = load_drugflow_model(DRUGFLOW_CKPT, device=DEVICE)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # 4. Process protein
    print("\n[4] Processing protein...")
    data, ref_size = process_protein_for_drugflow(PROTEIN_PDB, REF_LIGAND, model)
    print(f"  Reference ligand: {ref_size} atoms")

    from src.data.data_utils import TensorDict
    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(DEVICE),
        "pocket": TensorDict(**data["pocket"]).to(DEVICE),
    }

    # 5. Build Phase 1 guide
    print("\n[5] Building Phase 1 guide...")
    kts = KTSScheduler(alpha0=0.01, beta0=0.01)
    guide_fn = _Phase1GuideFn(
        site_energy=energy_fn,
        lambda_guide=0.5,
        guidance_start=0.05,
        guidance_end=0.95,
        grad_clip=0.5,
        kts=kts,
    ).to(DEVICE)

    # 6. Run Phase 1 — single attempt, single sample
    print("\n[6] Running Phase 1 (minimal: 4 atoms, 1 sample, 30 steps)...")
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**3

    t_start = time.time()
    with torch.no_grad():
        rdmols, trajectories, _ = model.sample(
            protein_data,
            n_samples=1,
            timesteps=30,       # Minimal ODE steps
            num_nodes=4,        # Small fragment
            guide_log_prob=guide_fn,
        )
    elapsed = time.time() - t_start
    mem_after = torch.cuda.memory_allocated() / 1024**3
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  GPU memory: {mem_before:.2f} GB → {mem_after:.2f} GB (peak {mem_peak:.2f} GB)")

    # 7. Check occupancy
    print("\n[7] Checking occupancy...")
    valid_mols = [m for m in rdmols if m is not None]
    print(f"  Valid molecules: {len(valid_mols)}/{len(rdmols)}")

    if valid_mols:
        mol = valid_mols[0]
        from guidance.two_stage_generation import _tensors_from_rdmol
        x, h = _tensors_from_rdmol(mol, device=DEVICE)

        if x is not None:
            diag = _compute_diagnostics(
                x, h, energy_fn, success_distance=2.5, min_compatibility=0.3
            )
            print(f"  Success: {diag['success']}")
            print(f"  Occupied sites: {diag['n_occupied_sites']}/{diag['n_sites']}")
            print(f"  Best distance: {diag['best_distance']:.2f} Å")
            print(f"  Best compat: {diag['best_compat']:.3f}")
            for ps in diag["per_site_best"]:
                print(f"    Site {ps['site_idx']}: d={ps['min_distance']:.2f}Å, "
                      f"compat={ps['best_compat']:.3f}, type={ps['best_atom_type']}, "
                      f"occupied={ps['occupied']}")

            # Extract anchors
            if diag["success"]:
                cfg = Phase1Config(success_distance=2.5, min_compatibility=0.3,
                                   anchor_selection="best_per_site")
                anchors = _extract_anchors(x, h, energy_fn, diag, cfg)
                if anchors:
                    print(f"\n  Anchor atoms: {anchors.n_anchors}")
                    for i in range(anchors.n_anchors):
                        atype = ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
                        pos = anchors.positions[i].cpu().tolist()
                        print(f"    Anchor {i}: {atype} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                              f"d={anchors.distances[i]:.2f}Å, compat={anchors.compat_scores[i]:.3f}")

    print("\n" + "=" * 60)
    print("Phase 1 Minimal Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
