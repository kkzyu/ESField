#!/usr/bin/env python3
"""
Unified Experiment Runner — all guidance strategies, one script.

Usage:
    # Main comparison matrix
    python run_unified_experiment.py --pocket 3mfw --guidance kag --n-samples 50

    # KAG ablation A: projection modes
    python run_unified_experiment.py --pocket 3mfw --guidance kag \
        --skip-phase1 --projection-mode internal --n-samples 50

    # KAG ablation B: single vs two-stage
    python run_unified_experiment.py --pocket 3mfw --guidance kag \
        --skip-phase1 --projection-mode com --n-samples 50

    # KAG ablation C: schedule
    python run_unified_experiment.py --pocket 3mfw --guidance kag \
        --schedule-type constant --n-samples 50
"""

from __future__ import annotations

import argparse, json, os, sys, time, warnings
from pathlib import Path
from functools import partial

import numpy as np
import torch

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

from guidance_unified import create_guidance
from guidance.latent_guidance import SiteCompatibilityEnergy

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

POCKET_CONFIG = {
    "3mfw": {"year": "2001-2010", "ref_atoms": 26, "n_hew": 7},
    "2gni": {"year": "2001-2010", "ref_atoms": 20, "n_hew": 4},
    "6o4x": {"year": "2011-2019", "ref_atoms": 22, "n_hew": 6},
    "2jke": {"year": "2001-2010", "ref_atoms": 24, "n_hew": 8},
    "2gqn": {"year": "2001-2010", "ref_atoms": 18, "n_hew": 10},
    "6phx": {"year": "2011-2019", "ref_atoms": 21, "n_hew": 8},
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
        "protein_pdb": str(protein_pdb),
        "ref_ligand": str(ref_ligand),
        "site_map": str(site_map) if site_map.exists() else None,
        "ref_atoms": cfg["ref_atoms"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Site energy builder
# ═══════════════════════════════════════════════════════════════════════════

def build_site_energy(site_map_path: str | None) -> SiteCompatibilityEnergy | None:
    if site_map_path is None:
        return None
    with open(site_map_path) as f:
        sm = json.load(f)
    hew_sites = [s for s in sm.get("sites", [])
                 if s.get("site_type") == "high_energy_water"]
    if not hew_sites:
        print("  WARNING: No HEW sites in site map")
        return None

    centers = torch.tensor([s["center"] for s in hew_sites], dtype=torch.float32)

    # Microenvironment classification
    def classify_env(features):
        hb = features.get("hbond_count", 0)
        hc = features.get("hydrophobic_contact_count", 0)
        if hc >= 3 and hb <= 1:
            return 0  # hydrophobic
        elif hb <= 1 and hc < 3:
            return 1  # polar_unsatisfied
        else:
            return 2  # mixed

    env_indices = torch.tensor([classify_env(s.get("features", {})) for s in hew_sites],
                                dtype=torch.long)

    site_energy = SiteCompatibilityEnergy(sigma_distance=3.0)
    site_energy.register_sites(centers, env_indices)
    return site_energy


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified guidance experiment runner")
    # Required
    parser.add_argument("--guidance", required=True,
                        choices=["unguided", "hard_fix", "lai_soft_fix",
                                 "badger_proxy", "kag"])
    parser.add_argument("--pocket", default="3mfw")

    # Generation
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Guidance hyperparams
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--lambda-ff", type=float, default=0.5)
    parser.add_argument("--lambda-site", type=float, default=0.5)
    parser.add_argument("--lambda-badger", type=float, default=1.0)
    parser.add_argument("--phase-gate", type=float, default=0.6)
    parser.add_argument("--schedule", default="quadratic")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # KAG-specific ablation flags
    parser.add_argument("--projection-mode", default="com",
                        choices=["com", "full", "internal"])
    parser.add_argument("--skip-phase1", action="store_true", default=False)
    parser.add_argument("--schedule-type", default="quadratic",
                        choices=["quadratic", "constant"])

    # Output
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    drugflow_utils.set_deterministic(seed=args.seed)
    drugflow_utils.disable_rdkit_logging()

    paths = get_pocket_paths(args.pocket)
    protein_pdb = paths["protein_pdb"]
    ref_ligand = paths["ref_ligand"]
    site_map = paths["site_map"]
    ref_atoms = paths["ref_atoms"]

    # Output dir naming including ablation flags
    suffix = ""
    if args.guidance == "kag":
        parts = [args.projection_mode]
        if args.skip_phase1:
            parts.append("single_stage")
        else:
            parts.append("two_stage")
        parts.append(args.schedule_type)
        suffix = "_" + "_".join(parts)
    elif args.skip_phase1:
        suffix = "_single_stage"

    if args.output_dir is None:
        output_dir = (ROOT / "experiments/master_experiments/unified"
                      / f"{args.pocket}_{args.guidance}{suffix}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = output_dir / "sdfs"
    sdf_dir.mkdir(exist_ok=True)

    print(f"=== Unified Experiment: {args.guidance} ===")
    print(f"  Pocket: {args.pocket} ({ref_atoms} ref atoms)")
    print(f"  N={args.n_samples}, steps={args.n_steps}")
    if args.guidance == "kag":
        print(f"  KAG: projection={args.projection_mode}, "
              f"skip_phase1={args.skip_phase1}, schedule={args.schedule_type}")

    # Load DrugFlow
    import argparse as _ap, pathlib as _pl, collections as _col
    torch.serialization.add_safe_globals([
        _ap.Namespace, _pl.PosixPath, _pl.WindowsPath,
        _pl.PurePosixPath, _pl.PureWindowsPath, _col.OrderedDict,
    ])
    model = DrugFlow.load_from_checkpoint(str(DRUGFLOW_CKPT),
                                           map_location=args.device, strict=False)
    model.datadir = str(DRUGFLOW_DIR / "src" / "default")
    model.setup(stage="generation")
    model.batch_size = model.eval_batch_size = args.batch_size
    model.eval().to(args.device)
    if args.n_steps is not None:
        model.T = args.n_steps
    print(f"  Model ready, T={model.T}")

    # Parse protein
    pdb_parser = PDBParser(QUIET=True)
    pdb_model = pdb_parser.get_structure("", protein_pdb)[0]
    rdmol = Chem.SDMolSupplier(ref_ligand)[0]
    if rdmol is None:
        mol2 = ref_ligand.replace(".sdf", ".mol2")
        rdmol = Chem.MolFromMol2File(mol2)
    if rdmol is None:
        raise RuntimeError(f"Cannot read: {ref_ligand}")

    ligand, pocket = process_raw_pair(
        pdb_model, rdmol, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=protein_pdb if model.dynamics.add_nma_feat else None)
    ligand["name"] = "ligand"

    # Build guidance
    guidance_kwargs = dict(
        total_steps=args.n_steps,
        phase_gate=args.phase_gate,
        schedule=args.schedule,
        device=args.device,
        verbose=True,
    )

    if args.guidance == "unguided":
        pass  # no extra args
    elif args.guidance == "hard_fix":
        # Build anchor target coordinates from reference ligand
        ref_conf = rdmol.GetConformer()
        anchor_indices = list(range(min(4, ref_atoms)))
        anchor_coords = torch.tensor([
            ref_conf.GetAtomPosition(i) for i in anchor_indices
        ], dtype=torch.float32)
        guidance_kwargs.update(
            anchor_indices=anchor_indices,
            anchor_coords=anchor_coords,
        )
    elif args.guidance == "lai_soft_fix":
        site_energy = build_site_energy(site_map)
        anchor_indices = list(range(min(4, ref_atoms)))
        guidance_kwargs.update(
            n_atoms=ref_atoms,
            site_map_path=site_map,
            anchor_indices=anchor_indices,
            lambda_ff=args.lambda_ff,
            lambda_site=args.lambda_site,
            grad_clip=args.grad_clip,
        )
    elif args.guidance == "badger_proxy":
        guidance_kwargs.update(
            n_atoms=ref_atoms,
            protein_pdb=protein_pdb,
            lambda_badger=args.lambda_badger,
            grad_clip=args.grad_clip,
        )
    elif args.guidance == "kag":
        site_energy = build_site_energy(site_map)
        anchor_indices = list(range(min(4, ref_atoms)))
        if site_energy is not None:
            site_energy.to(args.device)
        guidance_kwargs.update(
            anchor_indices=anchor_indices,
            site_energy=site_energy,
            projection_mode=args.projection_mode,
            skip_phase1=args.skip_phase1,
            schedule_type=args.schedule_type,
            lambda_max=args.lambda_max,
            grad_clip=args.grad_clip,
        )

    cb = create_guidance(args.guidance, **guidance_kwargs)

    # Generate
    n_generated = 0
    molecule_size = ref_atoms
    print(f"  Generating {args.n_samples} molecules ({args.guidance})...")
    t_start = time.time()

    with torch.no_grad():
        while n_generated < args.n_samples:
            dataset = [{"ligand": ligand, "pocket": pocket}
                       for _ in range(args.batch_size)]
            dataloader = DataLoader(
                dataset=dataset, batch_size=args.batch_size,
                collate_fn=partial(ProcessedLigandPocketDataset.collate_fn,
                                   ligand_transform=None),
                pin_memory=True)

            for data in dataloader:
                new_data = {
                    "ligand": TensorDict(**data["ligand"]).to(args.device),
                    "pocket": TensorDict(**data["pocket"]).to(args.device),
                }

                rdmols, rdpockets, _ = model.sample(
                    new_data, n_samples=1, timesteps=args.n_steps,
                    num_nodes=molecule_size, post_step_callback=cb)

                for mol in rdmols:
                    if mol is not None and n_generated < args.n_samples:
                        sdf_path = sdf_dir / f"mol_{n_generated:03d}.sdf"
                        try:
                            w = Chem.SDWriter(str(sdf_path))
                            try:
                                w.write(mol)
                            finally:
                                w.close()
                            n_generated += 1
                        except Exception:
                            # Kekulization failed — try sanitizing
                            try:
                                Chem.SanitizeMol(mol)
                                w = Chem.SDWriter(str(sdf_path))
                                try:
                                    w.write(mol)
                                finally:
                                    w.close()
                                n_generated += 1
                            except Exception:
                                pass  # skip invalid molecule
                    elif n_generated >= args.n_samples:
                        break

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.0f}s — {n_generated}/{args.n_samples} molecules")

    # Save metadata
    meta = {
        "pocket": args.pocket,
        "guidance": args.guidance,
        "n_samples": args.n_samples,
        "n_generated": n_generated,
        "n_steps": args.n_steps,
        "elapsed_s": elapsed,
        "guidance_summary": cb.get_summary(),
    }
    # Add KAG-specific ablation metadata
    if args.guidance == "kag":
        meta.update({
            "projection_mode": args.projection_mode,
            "skip_phase1": args.skip_phase1,
            "schedule_type": args.schedule_type,
        })

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  Output: {output_dir}")
    return output_dir, meta


if __name__ == "__main__":
    main()
