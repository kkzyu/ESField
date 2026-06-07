#!/usr/bin/env python3
"""Optimized v7 Phase 1 — stronger guidance, more steps, more samples.

Run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v7_phase1_optimized.py [--lambda L] [--steps S]
"""

import json, os, sys, time, warnings, argparse
from pathlib import Path

DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"
sys.path.insert(0, os.path.join(DRUGFLOW_DIR, "src"))
sys.path.insert(0, DRUGFLOW_DIR)

from src.model import lightning as lmod
from src.data.data_utils import process_raw_pair, TensorDict
from src.data.dataset import ProcessedLigandPocketDataset
from torch.utils.data import DataLoader
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

ESFIELD_ROOT = "/root/ESField"
for p in [f"{ESFIELD_ROOT}/src", ESFIELD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from guidance.latent_guidance import (
    build_site_energy_from_map, classify_hew_environment,
    ATOM_TYPE_VOCAB, ATOM_TYPE_TO_IDX,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    Phase1Config, _Phase1GuideFn, _compute_diagnostics, _extract_anchors,
    _tensors_from_rdmol,
)

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
SITE_MAP = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/site_maps/2g63_site_map.json"
PROTEIN_PDB = "/root/autodl-tmp/data/PDB/P-L/2001-2010/2g63/2g63_protein.pdb"
REF_LIGAND = "/root/autodl-tmp/data/PDB/P-L/2001-2010/2g63/2g63_ligand.sdf"
DEVICE = "cuda:0"


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
    from Bio.PDB import PDBParser
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
        [{"ligand": ligand_raw, "pocket": pocket_raw}], batch_size=1, collate_fn=collate)))
    return data, ref_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-early", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-attempts", type=int, default=3)
    parser.add_argument("--n-per-attempt", type=int, default=5)
    parser.add_argument("--n-atoms", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print(f"v7 Phase 1 — Optimized (λ={args.lambda_early}, steps={args.steps})")
    print("=" * 60)

    # 1. Site map
    site_map = json.load(open(SITE_MAP))
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    print(f"\n[1] {len(hew_sites)} HEW sites on 2g63")

    # 2. Site energy (all HEW, no filtering)
    energy_fn = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(DEVICE)
    print(f"[2] {energy_fn.n_sites} HEW sites registered")

    # 3. Load DrugFlow
    print("[3] Loading DrugFlow...")
    model = load_model(DRUGFLOW_CKPT, device=DEVICE)
    print(f"    {sum(p.numel() for p in model.parameters()):,} params, "
          f"GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 4. Process protein
    data, ref_size = process_protein(PROTEIN_PDB, REF_LIGAND, model)
    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(DEVICE),
        "pocket": TensorDict(**data["pocket"]).to(DEVICE),
    }
    print(f"[4] Protein processed (ref ligand: {ref_size} atoms)")

    # 5. Phase 1 guide with stronger settings
    kts = KTSScheduler(alpha0=0.01, beta0=0.01)
    guide_fn = _Phase1GuideFn(
        site_energy=energy_fn, lambda_guide=args.lambda_early,
        guidance_start=0.05, guidance_end=0.95,
        grad_clip=1.0, kts=kts,
    ).to(DEVICE)

    # 6. Run Phase 1
    print(f"\n[6] Phase 1: {args.n_attempts} attempts × {args.n_per_attempt} samples "
          f"× {args.n_atoms} atoms × {args.steps} steps, λ={args.lambda_early}")

    best_overall_d = float("inf")
    for attempt in range(args.n_attempts):
        t_start = time.time()
        with torch.no_grad():
            rdmols, _, _ = model.sample(
                protein_data, n_samples=args.n_per_attempt,
                timesteps=args.steps, num_nodes=args.n_atoms,
                guide_log_prob=guide_fn,
            )
        elapsed = time.time() - t_start

        best_attempt_d = float("inf")
        successes = []
        for idx, mol in enumerate(rdmols):
            if mol is None:
                continue
            x, h = _tensors_from_rdmol(mol, device=DEVICE)
            if x is None:
                continue
            diag = _compute_diagnostics(x, h, energy_fn, 2.5, 0.3)

            if diag["best_distance"] < best_attempt_d:
                best_attempt_d = diag["best_distance"]

            status = "✓" if diag["success"] else " "
            info = (f"  [{status}] Mol {idx}: d_min={diag['best_distance']:.2f}Å "
                    f"compat={diag['best_compat']:.3f} "
                    f"sites={diag['n_occupied_sites']}/{diag['n_sites']}")

            if diag["success"]:
                cfg = Phase1Config(success_distance=2.5, min_compatibility=0.3,
                                   anchor_selection="best_per_site")
                anchors = _extract_anchors(x, h, energy_fn, diag, cfg)
                if anchors:
                    types_str = ",".join(
                        f"{ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]}@"
                        f"{anchors.distances[i]:.2f}Å"
                        for i in range(anchors.n_anchors)
                    )
                    info += f"  anchors=[{types_str}]"
                    successes.append(anchors)

            if diag["success"] or diag["best_distance"] < 4.0:
                print(info)

        if best_attempt_d < best_overall_d:
            best_overall_d = best_attempt_d

        print(f"  Attempt {attempt+1}: {elapsed:.1f}s, "
              f"best_d={best_attempt_d:.2f}Å, "
              f"successes={len(successes)}/{args.n_per_attempt}")

        if successes:
            print(f"\n  ★★★ Phase 1 SUCCESS on attempt {attempt+1}! ★★★")
            print(f"  Anchor atoms ready for Phase 2.")
            # Save first successful anchors
            best_anchors = successes[0]
            anchor_dict = best_anchors.to_dict()
            out_path = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v7/phase1_anchors_2g63.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            json.dump(anchor_dict, open(out_path, "w"), indent=2)
            print(f"  Saved anchors to {out_path}")
            break
    else:
        print(f"\n  All attempts completed. Best distance overall: {best_overall_d:.2f}Å")
        print(f"  (Threshold for success: 2.5Å with compat ≥ 0.3)")

    print(f"\n{'='*60}")
    print(f"Phase 1: {'SUCCESS' if successes else 'NO SUCCESS (closest: ' + f'{best_overall_d:.2f}Å)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
