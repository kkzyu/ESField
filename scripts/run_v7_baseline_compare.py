#!/usr/bin/env python3
"""Compare v7 two-stage vs unconditional baseline on 2g63.

Run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v7_baseline_compare.py
"""

import json, os, sys, time, warnings
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
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

ESFIELD_ROOT = "/root/ESField"
for p in [f"{ESFIELD_ROOT}/src", ESFIELD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from guidance.latent_guidance import build_site_energy_from_map
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    TwoStageGuideFn, AnchorAtoms, Phase2Config, _tensors_from_rdmol,
    _compute_diagnostics,
)
from evaluation.site_occupancy import site_occupancy_summary
from evaluation.posu import compute_posu

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
SITE_MAP = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/site_maps/2g63_site_map.json"
PROTEIN_PDB = "/root/autodl-tmp/data/PDB/P-L/2001-2010/2g63/2g63_protein.pdb"
REF_LIGAND = "/root/autodl-tmp/data/PDB/P-L/2001-2010/2g63/2g63_ligand.sdf"
ANCHOR_FILE = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v7/phase1_anchors_2g63.json"
OUTPUT_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v7"
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


def generate_and_evaluate(model, protein_data, site_map, label, guide_fn=None,
                          n_samples=5, num_nodes=29, timesteps=100):
    """Generate molecules and compute metrics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(
            protein_data, n_samples=n_samples, timesteps=timesteps,
            num_nodes=num_nodes, guide_log_prob=guide_fn,
        )
    elapsed = time.time() - t0

    valid = [m for m in rdmols if m is not None]
    print(f"  Valid: {len(valid)}/{n_samples}, Time: {elapsed:.1f}s")

    # Quality
    qeds, mws, logps = [], [], []
    for m in valid:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            qeds.append(QED.qed(m))
            mws.append(Descriptors.MolWt(m))
            logps.append(Descriptors.MolLogP(m))
        except Exception:
            pass

    # Site occupancy
    occ = site_occupancy_summary(valid, site_map, threshold=2.5)
    dor = occ["direct_occupancy"]["rate"]
    bcd_mean = occ["compatible_distance"]["mean"]
    bcd_min = occ["compatible_distance"]["min"]
    sites_occ = occ["compatible_distance"]["n_sites_occupied"]

    # POSU
    posu_vals = []
    for m in valid:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            p = compute_posu(m, site_map)
            posu_vals.append(p["posu"])
        except Exception:
            pass

    results = {
        "label": label,
        "valid": len(valid),
        "n_total": n_samples,
        "qed_mean": float(np.mean(qeds)) if qeds else 0,
        "mw_mean": float(np.mean(mws)) if mws else 0,
        "logp_mean": float(np.mean(logps)) if logps else 0,
        "direct_occ_rate": dor,
        "best_compat_d_mean": bcd_mean,
        "best_compat_d_min": bcd_min,
        "n_sites_occupied": sites_occ,
        "posu_mean": float(np.mean(posu_vals)) if posu_vals else 0,
        "elapsed": elapsed,
    }

    print(f"  QED: {results['qed_mean']:.3f}, MW: {results['mw_mean']:.0f}, "
          f"logP: {results['logp_mean']:.1f}")
    print(f"  DirectOcc: {dor:.3f} ({int(dor*len(valid))}/{len(valid)})")
    print(f"  BestCompatD: mean={bcd_mean:.2f}Å, min={bcd_min:.2f}Å")
    print(f"  Sites occupied: {sites_occ}/4")
    print(f"  POSU: {results['posu_mean']:.3f}")

    return results, valid


def main():
    print("=" * 60)
    print("v7 vs Baseline Comparison — 2g63")
    print("=" * 60)

    site_map = json.load(open(SITE_MAP))
    energy_fn = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(DEVICE)

    print("[1] Loading DrugFlow...")
    model = load_model(DRUGFLOW_CKPT, device=DEVICE)

    print("[2] Processing protein...")
    data, ref_size = process_protein(PROTEIN_PDB, REF_LIGAND, model)
    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(DEVICE),
        "pocket": TensorDict(**data["pocket"]).to(DEVICE),
    }

    all_results = {}

    # ---- Condition 1: Unconditional baseline ----
    r1, _ = generate_and_evaluate(
        model, protein_data, site_map, "BASELINE (unconditional)",
        guide_fn=None, n_samples=10,
    )
    all_results["baseline"] = r1

    # ---- Condition 2: v7 two-stage (current anchors) ----
    anchors_data = json.load(open(ANCHOR_FILE))
    anchors = AnchorAtoms(
        positions=torch.tensor(anchors_data["positions"], dtype=torch.float32, device=DEVICE),
        type_indices=torch.tensor(anchors_data["type_indices"], dtype=torch.long, device=DEVICE),
        type_probs=torch.ones(anchors_data["n_anchors"], 11, device=DEVICE) / 11,
        occupied_sites=anchors_data["occupied_sites"],
        compat_scores=anchors_data["compat_scores"],
        distances=anchors_data["distances"],
    )

    kts = KTSScheduler(alpha0=0.005, beta0=0.01)
    cfg = Phase2Config(lambda_late=0.1, restraint_force=10.0,
                       guidance_start=0.1, guidance_end=0.9, grad_clip=0.3)
    guide = TwoStageGuideFn(energy_fn, anchors, cfg, kts).to(DEVICE)
    guide.set_anchor_indices([0], 29)

    r2, _ = generate_and_evaluate(
        model, protein_data, site_map, "v7 Two-Stage (k=10, λ=0.1)",
        guide_fn=guide, n_samples=10,
    )
    all_results["v7_weak"] = r2

    # ---- Condition 3: v7 with stronger restraint ----
    cfg3 = Phase2Config(lambda_late=0.1, restraint_force=100.0,
                        guidance_start=0.1, guidance_end=0.9, grad_clip=0.3)
    guide3 = TwoStageGuideFn(energy_fn, anchors, cfg3, kts).to(DEVICE)
    guide3.set_anchor_indices([0], 29)

    r3, _ = generate_and_evaluate(
        model, protein_data, site_map, "v7 Two-Stage (k=100, λ=0.1)",
        guide_fn=guide3, n_samples=10,
    )
    all_results["v7_strong"] = r3

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Condition':<30} {'QED':>6} {'MW':>6} {'OccRate':>8} {'BestD':>7} {'POSU':>6}")
    print("-" * 70)
    for key, r in all_results.items():
        print(f"{r['label']:<30} {r['qed_mean']:>6.3f} {r['mw_mean']:>6.0f} "
              f"{r['direct_occ_rate']:>8.3f} {r['best_compat_d_min']:>6.2f}Å "
              f"{r['posu_mean']:>6.3f}")

    # Save
    out = {k: {kk: vv for kk, vv in v.items() if kk != 'label'}
           for k, v in all_results.items()}
    for k, v in all_results.items():
        out[k]["label"] = v["label"]
    json.dump(out, open(f"{OUTPUT_DIR}/v7_comparison_2g63.json", "w"), indent=2, default=str)
    print(f"\nSaved to {OUTPUT_DIR}/v7_comparison_2g63.json")


if __name__ == "__main__":
    main()
