#!/usr/bin/env python3
"""Standalone v7 Phase 1 test — DrugFlow import first, then ESField.

Must be run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v7_phase1_standalone.py
"""

import json, os, sys, time, warnings
from pathlib import Path

# ── Step 0: Import DrugFlow FIRST (before ESField pollutes the 'src' namespace) ──
# DrugFlow's src/ has NO __init__.py. ESField's src/ HAS __init__.py.
# If ESField's src is on sys.path first, Python won't find DrugFlow's src.model.

# Ensure DrugFlow's src is findable
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

# ── Step 1: NOW add ESField to path ──
ESFIELD_ROOT = "/root/ESField"
for p in [f"{ESFIELD_ROOT}/src", ESFIELD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from guidance.latent_guidance import (
    SiteCompatibilityEnergy, build_site_energy_from_map,
    classify_hew_environment, ATOM_TYPE_VOCAB, ATOM_TYPE_TO_IDX,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    Phase1Config, _Phase1GuideFn, _compute_diagnostics, _extract_anchors,
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
        [{"ligand": ligand_raw, "pocket": pocket_raw}],
        batch_size=1, collate_fn=collate)))
    return data, ref_size


def main():
    print("=" * 60)
    print("v7 Phase 1 — Standalone GPU Test (2g63)")
    print("=" * 60)

    # 1. Load site map
    print("\n[1] Loading site map...")
    site_map = json.load(open(SITE_MAP))
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    for s in hew_sites:
        env = classify_hew_environment(s)
        print(f"  Site {s['site_id']}: {env}, conf={s['confidence']:.2f}")
    print(f"  Total: {len(hew_sites)} HEW sites")

    # 2. Build site energy (top-2 HEW)
    print("\n[2] Building SiteCompatibilityEnergy...")
    energy_fn = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
        top_k=2,
    ).to(DEVICE)
    print(f"  Registered {energy_fn.n_sites} HEW site(s)")

    # 3. Load DrugFlow
    print("\n[3] Loading DrugFlow model...")
    t0 = time.time()
    model = load_model(DRUGFLOW_CKPT, device=DEVICE)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB total")

    # 4. Process protein
    print("\n[4] Processing protein...")
    data, ref_size = process_protein(PROTEIN_PDB, REF_LIGAND, model)
    print(f"  Reference ligand: {ref_size} atoms")
    print(f"  GPU memory after protein load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(DEVICE),
        "pocket": TensorDict(**data["pocket"]).to(DEVICE),
    }
    print(f"  GPU memory after to(device): {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 5. Phase 1 guide
    print("\n[5] Phase 1 guide...")
    kts = KTSScheduler(alpha0=0.01, beta0=0.01)
    guide_fn = _Phase1GuideFn(
        site_energy=energy_fn, lambda_guide=0.5,
        guidance_start=0.05, guidance_end=0.95,
        grad_clip=0.5, kts=kts,
    ).to(DEVICE)

    # 6. Run Phase 1 — multiple attempts
    print("\n[6] Running Phase 1...")
    n_attempts = 3
    n_per_attempt = 3

    for attempt in range(n_attempts):
        print(f"\n  --- Attempt {attempt+1}/{n_attempts} ---")
        torch.cuda.reset_peak_memory_stats()

        t_start = time.time()
        with torch.no_grad():
            rdmols, trajectories, _ = model.sample(
                protein_data, n_samples=n_per_attempt, timesteps=30,
                num_nodes=4, guide_log_prob=guide_fn,
            )
        elapsed = time.time() - t_start
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Time: {elapsed:.1f}s, Peak GPU: {mem_peak:.2f} GB")

        from guidance.two_stage_generation import _tensors_from_rdmol

        best_d = float("inf")
        for idx, mol in enumerate(rdmols):
            if mol is None:
                continue
            x, h = _tensors_from_rdmol(mol, device=DEVICE)
            if x is None:
                continue
            diag = _compute_diagnostics(x, h, energy_fn, 2.5, 0.3)
            status = "OCCUPIED" if diag["success"] else "missed"
            if diag["best_distance"] < best_d:
                best_d = diag["best_distance"]

            type_str = ",".join(
                f"{ps['best_atom_type']}@{ps['min_distance']:.1f}Å"
                for ps in diag["per_site_best"]
            )
            print(f"    Mol {idx}: {status} | d_min={diag['best_distance']:.2f}Å | "
                  f"compat={diag['best_compat']:.3f} | [{type_str}]")

            if diag["success"]:
                # Extract anchors
                cfg = Phase1Config(success_distance=2.5, min_compatibility=0.3,
                                   anchor_selection="best_per_site")
                anchors = _extract_anchors(x, h, energy_fn, diag, cfg)
                if anchors:
                    print(f"      ★ Anchors: {anchors.n_anchors} atom(s)")
                    for i in range(anchors.n_anchors):
                        atype = ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
                        pos = anchors.positions[i].cpu().tolist()
                        print(f"        {atype} at ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
                              f"d={anchors.distances[i]:.2f}Å compat={anchors.compat_scores[i]:.3f}")

        # Check if any molecule succeeded
        any_success = any(
            _compute_diagnostics(*[t.to(DEVICE) for t in _tensors_from_rdmol(m, device=DEVICE)],
                                 energy_fn, 2.5, 0.3)["success"]
            for m in rdmols if m is not None
        )
        if any_success:
            print(f"\n  ✓ Phase 1 SUCCESS on attempt {attempt+1}!")
            break
    else:
        print(f"\n  ✗ All {n_attempts} attempts failed. Best distance: {best_d:.2f}Å")

    print(f"\n{'='*60}")
    print("Phase 1 Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
