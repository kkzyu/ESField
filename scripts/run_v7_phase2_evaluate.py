#!/usr/bin/env python3
"""v7 Phase 2 + Evaluation — full molecule generation with anchor restraint.

Run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v7_phase2_evaluate.py
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
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

ESFIELD_ROOT = "/root/ESField"
for p in [f"{ESFIELD_ROOT}/src", ESFIELD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from guidance.latent_guidance import (
    build_site_energy_from_map, classify_hew_environment,
    SiteCompatibilityEnergy, ATOM_TYPE_VOCAB,
)
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


def load_anchors(path, device="cuda:0"):
    """Load saved anchors from Phase 1 JSON."""
    data = json.load(open(path))
    n_types = 11  # ATOM_TYPE_VOCAB length
    if "type_probs" in data:
        type_probs = torch.tensor(data["type_probs"], dtype=torch.float32, device=device)
    else:
        # Backward compat: create one-hot from type_indices
        type_probs = torch.zeros(data["n_anchors"], n_types, device=device)
        for i, tidx in enumerate(data["type_indices"]):
            type_probs[i, tidx] = 1.0
    return AnchorAtoms(
        positions=torch.tensor(data["positions"], dtype=torch.float32, device=device),
        type_indices=torch.tensor(data["type_indices"], dtype=torch.long, device=device),
        type_probs=type_probs,
        occupied_sites=data["occupied_sites"],
        compat_scores=data["compat_scores"],
        distances=data["distances"],
    )


def save_mols(rdmols, path):
    """Save RDKit molecules to SDF."""
    writer = Chem.SDWriter(path)
    writer.SetKekulize(False)
    for m in rdmols:
        if m is not None:
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass
            writer.write(m)
    writer.close()


def main():
    print("=" * 60)
    print("v7 Phase 2 + Evaluation — 2g63")
    print("=" * 60)

    # 1. Load anchors
    print("\n[1] Loading Phase 1 anchors...")
    anchors = load_anchors(ANCHOR_FILE, device=DEVICE)
    print(f"  {anchors.n_anchors} anchor(s)")
    for i in range(anchors.n_anchors):
        atype = ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
        pos = anchors.positions[i].cpu().tolist()
        print(f"  Anchor {i}: {atype} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        print(f"    occupied_site={anchors.occupied_sites[i]}, "
              f"compat={anchors.compat_scores[i]:.3f}")

    # 2. Site energy for Phase 2
    site_map = json.load(open(SITE_MAP))
    energy_fn = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(DEVICE)
    print(f"\n[2] {energy_fn.n_sites} HEW sites for Phase 2 guidance")

    # 3. DrugFlow
    print("\n[3] Loading DrugFlow...")
    model = load_model(DRUGFLOW_CKPT, device=DEVICE)
    print(f"    GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 4. Protein
    data, ref_size = process_protein(PROTEIN_PDB, REF_LIGAND, model)
    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(DEVICE),
        "pocket": TensorDict(**data["pocket"]).to(DEVICE),
    }
    full_mol_size = ref_size  # Match reference ligand size
    print(f"\n[4] Protein processed, target molecule size: {full_mol_size} atoms")

    # 5. Phase 2 guide
    kts = KTSScheduler(alpha0=0.005, beta0=0.01)
    phase2_config = Phase2Config(
        fix_atoms=True, restraint_force=10.0,
        lambda_late=0.1, guidance_start=0.1, guidance_end=0.90,
        grad_clip=0.3,
    )
    guide_fn = TwoStageGuideFn(
        site_energy=energy_fn, anchors=anchors, config=phase2_config, kts=kts,
    ).to(DEVICE)
    guide_fn.set_anchor_indices(list(range(anchors.n_anchors)), full_mol_size)

    # 6. Phase 2: generate full molecules
    n_samples = 5
    timesteps = 100
    print(f"\n[6] Phase 2: {n_samples} molecules × {full_mol_size} atoms × {timesteps} steps")

    t_start = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(
            protein_data, n_samples=n_samples, timesteps=timesteps,
            num_nodes=full_mol_size, guide_log_prob=guide_fn,
        )
    elapsed = time.time() - t_start
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"    Time: {elapsed:.1f}s, GPU: {mem:.2f} GB")

    valid = [m for m in rdmols if m is not None]
    print(f"    Valid: {len(valid)}/{n_samples}")

    # 7. Save molecules
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sdf_path = f"{OUTPUT_DIR}/v7_phase2_2g63.sdf"
    save_mols(valid, sdf_path)
    print(f"\n[7] Saved {len(valid)} molecules to {sdf_path}")

    # 8. Quality metrics
    print(f"\n[8] Quality metrics:")
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

    if qeds:
        print(f"    QED:  {np.mean(qeds):.3f} ± {np.std(qeds):.3f}")
        print(f"    MW:   {np.mean(mws):.0f} ± {np.std(mws):.0f}")
        print(f"    logP: {np.mean(logps):.1f} ± {np.std(logps):.1f}")

    # 9. Site occupancy
    print(f"\n[9] Site occupancy metrics:")
    occ = site_occupancy_summary(valid, site_map, threshold=2.5)
    dor = occ["direct_occupancy"]
    bcd = occ["compatible_distance"]
    print(f"    Direct occupancy rate: {dor['rate']:.3f} ({dor['n_occupied']}/{dor['n_total']})")
    print(f"    Best compatible distance: mean={bcd['mean']:.2f}Å, min={bcd['min']:.2f}Å")
    print(f"    Sites occupied (d≤2.5Å): {bcd['n_sites_occupied']}/{bcd['n_sites_total']}")

    # Also check anchor preservation
    print(f"\n[10] Anchor preservation check:")
    for idx, mol in enumerate(valid):
        x, h = _tensors_from_rdmol(mol, device=DEVICE)
        if x is None:
            continue
        # Check anchor positions
        for a_idx in range(min(anchors.n_anchors, x.shape[0])):
            anchor_target = anchors.positions[a_idx]
            actual_pos = x[a_idx]
            drift = (actual_pos - anchor_target).norm().item()
            print(f"    Mol {idx} anchor {a_idx}: drift={drift:.2f}Å")

    # 11. POSU/HEWU
    print(f"\n[11] POSU-v2.1 / HEWU:")
    posu_results = []
    for m in valid:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            posu = compute_posu(m, site_map)
            posu_results.append(posu)
        except Exception:
            pass
    if posu_results:
        pu = [p["posu"] for p in posu_results]
        hu = [p["hew_mean"] for p in posu_results]
        print(f"    POSU: {np.mean(pu):.3f} ± {np.std(pu):.3f}")
        print(f"    HEWU: {np.mean(hu):.3f} ± {np.std(hu):.3f}")

    # Save metrics
    metrics = {
        "pocket": "2g63",
        "n_samples": n_samples,
        "valid": len(valid),
        "qed_mean": float(np.mean(qeds)) if qeds else 0,
        "qed_std": float(np.std(qeds)) if qeds else 0,
        "mw_mean": float(np.mean(mws)) if mws else 0,
        "logp_mean": float(np.mean(logps)) if logps else 0,
        "direct_occupancy_rate": dor["rate"],
        "best_compatible_distance_mean": bcd["mean"],
        "best_compatible_distance_min": bcd["min"],
        "n_sites_occupied": bcd["n_sites_occupied"],
        "n_sites_total": bcd["n_sites_total"],
        "posu_mean": float(np.mean(pu)) if posu_results else 0,
        "hewu_mean": float(np.mean(hu)) if posu_results else 0,
        "elapsed": elapsed,
        "gpu_memory_gb": mem,
    }
    metrics_path = f"{OUTPUT_DIR}/v7_phase2_metrics_2g63.json"
    json.dump(metrics, open(metrics_path, "w"), indent=2, default=str)
    print(f"\n    Metrics saved to {metrics_path}")

    print(f"\n{'='*60}")
    print("Phase 2 + Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
