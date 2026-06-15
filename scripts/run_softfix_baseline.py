#!/usr/bin/env python3
"""
Lai + Soft-Fix Baseline Experiment.

Runs the SoftFixGuidance callback (global MMFF94 gradient + HEW harmonic
constraint) on the DrugFlow backbone for the 3mfw pocket, N=50.

Expected outcome (vs. KAG kinematic anchoring):
  - DirectOcc_HEW: HIGH (harmonic constraint pulls anchors to HEW sites)
  - Strain/atom:    SIGNIFICANTLY HIGHER than KAG (R^{3N} injection)
  - ρ_KPE:          MODERATE (MMFF94 + harmonic gradients inject KE)
"""

from __future__ import annotations

import argparse, json, os, sys, time, warnings
from pathlib import Path
from typing import Optional
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASELINES = Path("/root/baselines")
DATA_ROOT = Path("/root/autodl-tmp/data")
DRUGFLOW_DIR = BASELINES / "DrugFlow/code/DrugFlow-main"
DRUGFLOW_CKPT = Path("/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt")

sys.path.insert(0, str(DRUGFLOW_DIR))
sys.path.insert(0, str(DRUGFLOW_DIR / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from Bio.PDB import PDBParser
from rdkit import Chem
from torch.utils.data import DataLoader

from src.data.dataset import ProcessedLigandPocketDataset
from src.data.data_utils import TensorDict, process_raw_pair
from src.model.lightning import DrugFlow
from src import utils as drugflow_utils

from naive_forcefield_guidance import SoftFixGuidance
from kpe_instrumentation import KPETracker, KPELogger

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

POCKET_CONFIG = {
    "3mfw": {"year": "2001-2010", "ref_atoms": 26, "n_hew": 7},
}

SITE_MAP_DIR = ROOT / "experiments/targetdiff_replication/site_maps"


def get_pocket_paths(pocket_id: str) -> dict:
    cfg = POCKET_CONFIG[pocket_id]
    year = cfg["year"]
    base = DATA_ROOT / "PDB/P-L" / year / pocket_id
    protein_pdb = base / f"{pocket_id}_protein.pdb"
    ref_ligand = base / f"{pocket_id}_ligand.sdf"
    site_map = SITE_MAP_DIR / f"{pocket_id}_site_map.json"
    return {
        "protein_pdb": protein_pdb,
        "ref_ligand": ref_ligand,
        "site_map": site_map if site_map.exists() else None,
        "ref_atoms": cfg["ref_atoms"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Lai + Soft-Fix Baseline")
    parser.add_argument("--pocket", default="3mfw")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--lambda-ff", type=float, default=0.5)
    parser.add_argument("--k-hew", type=float, default=1.0)
    parser.add_argument("--phase-gate", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schedule", default="quadratic")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    drugflow_utils.set_deterministic(seed=args.seed)
    drugflow_utils.disable_rdkit_logging()

    paths = get_pocket_paths(args.pocket)
    protein_pdb = paths["protein_pdb"]
    ref_ligand = paths["ref_ligand"]
    site_map = paths["site_map"]
    ref_atoms = paths["ref_atoms"]

    if args.output_dir is None:
        output_dir = ROOT / "experiments/master_experiments/softfix_baseline" / args.pocket
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = output_dir / "sdfs"
    sdf_dir.mkdir(exist_ok=True)

    print(f"=== Lai + Soft-Fix Baseline ===")
    print(f"  Pocket: {args.pocket}")
    print(f"  Protein: {protein_pdb}")
    print(f"  Ref ligand: {ref_ligand} ({ref_atoms} atoms)")
    print(f"  Site map: {site_map}")
    print(f"  λ_ff={args.lambda_ff}, k_hew={args.k_hew}, phase_gate={args.phase_gate}")
    print(f"  N={args.n_samples}, steps={args.n_steps}")
    print(f"  Output: {output_dir}")

    if not protein_pdb.exists():
        raise FileNotFoundError(f"Protein: {protein_pdb}")
    if not ref_ligand.exists():
        raise FileNotFoundError(f"Ligand: {ref_ligand}")
    if not site_map or not site_map.exists():
        raise FileNotFoundError(f"Site map: {site_map}")

    # Load HEW site centers for anchor assignment
    with open(site_map) as f:
        sm = json.load(f)
    hew_sites = [s for s in sm.get("sites", []) if s.get("site_type") == "high_energy_water"]
    print(f"  HEW sites: {len(hew_sites)}")

    # Anchor indices: use first 4 atoms as anchors (matching Phase 1 pattern)
    anchor_indices = [0, 1, 2, 3]

    print(f"  Anchor indices: {anchor_indices}")
    print(f"  Loading DrugFlow...")
    t0 = time.time()

    import argparse as _ap, pathlib as _pl, collections as _col
    torch.serialization.add_safe_globals([
        _ap.Namespace,
        _pl.PosixPath, _pl.WindowsPath, _pl.PurePosixPath, _pl.PureWindowsPath,
        _col.OrderedDict,
    ])
    model = DrugFlow.load_from_checkpoint(str(DRUGFLOW_CKPT), map_location=args.device, strict=False)
    model.datadir = str(DRUGFLOW_DIR / "src" / "default")
    model.setup(stage="generation")
    model.batch_size = model.eval_batch_size = args.batch_size
    model.eval().to(args.device)
    if args.n_steps is not None:
        model.T = args.n_steps
    print(f"  Model ready in {time.time()-t0:.1f}s, T={model.T}")

    pdb_parser = PDBParser(QUIET=True)
    pdb_model = pdb_parser.get_structure("", str(protein_pdb))[0]
    rdmol = Chem.SDMolSupplier(str(ref_ligand))[0]
    if rdmol is None:
        mol2 = str(ref_ligand).replace(".sdf", ".mol2")
        rdmol = Chem.MolFromMol2File(mol2)
    if rdmol is None:
        raise RuntimeError(f"Cannot read: {ref_ligand}")

    ligand, pocket = process_raw_pair(
        pdb_model, rdmol, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=str(protein_pdb) if model.dynamics.add_nma_feat else None)
    ligand["name"] = "ligand"

    kpe_logger = KPELogger(condition_name="soft_fix", pocket_name=args.pocket,
                           output_dir=str(output_dir / "kpe"))
    n_generated = 0
    molecule_size = ref_atoms

    print(f"  Generating {args.n_samples} molecules (strategy=soft_fix)...")
    with torch.no_grad():
        while n_generated < args.n_samples:
            dataset = [{"ligand": ligand, "pocket": pocket} for _ in range(args.batch_size)]
            dataloader = DataLoader(
                dataset=dataset, batch_size=args.batch_size,
                collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
                pin_memory=True)

            for data in dataloader:
                new_data = {
                    "ligand": TensorDict(**data["ligand"]).to(args.device),
                    "pocket": TensorDict(**data["pocket"]).to(args.device),
                }

                # Create SoftFix callback for this batch
                # Note: atom_types are not known at initialization; the callback
                # will use whatever 'h' the ligand dict has at call time.
                cb = SoftFixGuidance(
                    n_atoms=molecule_size,
                    atom_types=None,  # inferred from ligand['h'] at call time
                    lambda_ff=args.lambda_ff,
                    k_hew=args.k_hew,
                    site_map_path=str(site_map),
                    anchor_indices=anchor_indices,
                    total_steps=args.n_steps,
                    framework="ode",
                    schedule=args.schedule,
                    grad_clip=args.grad_clip,
                    phase_gate=args.phase_gate,
                    device=args.device,
                    verbose=(n_generated == 0),
                )

                rdmols, rdpockets, _ = model.sample(
                    new_data, n_samples=1, timesteps=args.n_steps,
                    num_nodes=molecule_size, post_step_callback=cb)

                for mol in rdmols:
                    if mol is not None and n_generated < args.n_samples:
                        sdf_path = sdf_dir / f"mol_{n_generated:03d}.sdf"
                        w = Chem.SDWriter(str(sdf_path))
                        try:
                            w.write(mol)
                        finally:
                            w.close()
                        n_generated += 1
                    elif n_generated >= args.n_samples:
                        break

    # Save metadata
    meta = {
        "pocket": args.pocket,
        "strategy": "soft_fix",
        "n_samples": args.n_samples,
        "n_generated": n_generated,
        "n_steps": args.n_steps,
        "lambda_ff": args.lambda_ff,
        "k_hew": args.k_hew,
        "phase_gate": args.phase_gate,
        "anchor_indices": anchor_indices,
        "n_hew_sites": len(hew_sites),
        "guidance_summary": cb.get_summary() if 'cb' in dir() else {},
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nDone! Generated {n_generated}/{args.n_samples} molecules")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
