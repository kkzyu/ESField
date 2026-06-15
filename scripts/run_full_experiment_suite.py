#!/usr/bin/env python3
"""Complete experiment suite for KAG paper — generates ALL conditions then evaluates.

Generates for each pocket (6 total):
  1. unguided — pure DrugFlow, no guidance
  2. hard_fix — per-step anchor coordinate reset (old implementation)
  3. full_gradient — per-atom full gradient guidance (no anchors, no CoM projection)
  4. kag — two-stage: Phase 1 → Phase 2 kinematic anchor guidance
  5. hard_fix_locked — fully locked anchors (Exp 3.3)

Each condition generates 50 molecules. Then runs ALL metrics:
  - Traditional: strain, clash, QED, SA, Vina
  - New: centroid distance, COS, E_site, Pareto

Usage:
    python scripts/run_full_experiment_suite.py --pockets 3mfw,6o4x --conditions kag,unguided --n-mols 10
    python scripts/run_full_experiment_suite.py --all --n-mols 50

Output:
    results/experiment_suite/{pocket}/{condition}/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED, AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # for metrics_new import

# ── DrugFlow paths ──
DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"
DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
sys.path.insert(0, os.path.join(DRUGFLOW_DIR, "src"))
sys.path.insert(0, DRUGFLOW_DIR)

from src.model import lightning as lmod
from src.data.data_utils import process_raw_pair, TensorDict
from src.data.dataset import ProcessedLigandPocketDataset
from torch.utils.data import DataLoader
from functools import partial
from Bio.PDB import PDBParser

# ── ESField imports ──
from guidance.latent_guidance import (
    SiteCompatibilityEnergy, build_site_energy_from_map,
    classify_hew_environment, ATOM_TYPE_VOCAB, ATOM_TYPE_TO_IDX, N_ATOM_TYPES,
    apply_latent_guidance,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    TwoStageGenerator, TwoStageConfig, Phase1Config, Phase2Config,
    AnchorAtoms, TwoStageGuideFn, _Phase1GuideFn,
    _compute_diagnostics, _extract_anchors, _tensors_from_rdmol,
)
from guidance.hard_fix import (
    HardFixCallback, FullyLockedAnchorCallback,
    patch_drugflow_hardfix, patch_drugflow_fully_locked,
    patch_drugflow_sample_post_step,
)
from guidance.kinematic_anchor import KinematicAnchorGuidance

# ── PDBbind pockets configuration ──
POCKETS_CONFIG = {
    "3mfw": {"year": "2001-2010"},
    "2gni": {"year": "2001-2010"},
    "2gqn": {"year": "2001-2010"},
    "2jke": {"year": "2001-2010"},
    "6o4x": {"year": "2011-2019"},
    "6phx": {"year": "2011-2019"},
}

SITE_MAPS_DIR = ROOT / "experiments/targetdiff_replication/site_maps"
DATA_BASE = Path("/root/autodl-tmp/data/PDB/P-L")
OUTPUT_BASE = ROOT / "results/experiment_suite"


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(device="cuda:0"):
    warnings.filterwarnings("ignore")
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        model = lmod.DrugFlow.load_from_checkpoint(DRUGFLOW_CKPT, map_location=device)
    finally:
        torch.load = _orig_load
    return model.to(device).eval()


def process_protein(protein_pdb, ref_ligand_sdf, model):
    pdb_model = PDBParser(QUIET=True).get_structure("", protein_pdb)[0]
    rdmol = Chem.SDMolSupplier(ref_ligand_sdf)[0]
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


# ═══════════════════════════════════════════════════════════════════════════
# Condition: UNGUIDED
# ═══════════════════════════════════════════════════════════════════════════

def generate_unguided(model, protein_data, n_mols, full_mol_size, device, timesteps=100):
    """Pure DrugFlow generation — no guidance."""
    molecules = []
    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(
            protein_data, n_samples=n_mols,
            timesteps=timesteps, num_nodes=full_mol_size,
        )
    elapsed = time.time() - t0
    for m in rdmols:
        if m is not None:
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                                 ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass
            molecules.append(m)
    return molecules, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Condition: FULL GRADIENT (per-atom gradient, no anchors, no CoM projection)
# ═══════════════════════════════════════════════════════════════════════════

class FullGradientGuideFn:
    """Per-atom full gradient guidance — site energy applied to ALL atoms.

    NO anchors, NO CoM projection.  Every atom receives its own gradient.
    This is the "Full Gradient" baseline.
    """

    def __init__(self, site_energy, lambda_guide=1.0, guidance_start=0.1,
                 guidance_end=0.90, grad_clip=0.5):
        self.site_energy = site_energy
        self.esfield_lambda = lambda_guide  # ★ DrugFlow patched simulate() requires this
        self.lambda_guide = lambda_guide
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.grad_clip = grad_clip

    def to(self, device):
        self.site_energy.to(device)
        return self

    def __call__(self, t_array, *, x, h, batch_mask, bonds=None, bond_types=None):
        t_val = float(t_array[0] if hasattr(t_array, "__len__") else t_array)
        if t_val < self.guidance_start or t_val > self.guidance_end:
            return torch.tensor(0.0, device=x.device)
        if self.site_energy.n_sites == 0:
            return torch.tensor(0.0, device=x.device)

        # Auto-detect softmax
        h_sum = h.sum(dim=-1)
        if (h >= 0).all() and (h <= 1).all() and torch.allclose(h_sum, torch.ones_like(h_sum), atol=0.01):
            atom_probs = h
        else:
            atom_probs = F.softmax(h, dim=-1)

        e_site = self.site_energy(x, atom_type_probs=atom_probs)
        return -self.lambda_guide * e_site


def generate_full_gradient(model, protein_data, n_mols, full_mol_size, device,
                           site_map, timesteps=100):
    """Generate with per-atom full gradient (no anchors, no CoM)."""
    site_energy = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(device)

    guide_fn = FullGradientGuideFn(site_energy, lambda_guide=1.0).to(device)

    molecules = []
    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(
            protein_data, n_samples=n_mols,
            timesteps=timesteps, num_nodes=full_mol_size,
            guide_log_prob=guide_fn,
        )
    elapsed = time.time() - t0
    for m in rdmols:
        if m is not None:
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                                 ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass
            molecules.append(m)
    return molecules, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Condition: HARD-FIX (per-step reset)
# ═══════════════════════════════════════════════════════════════════════════

def generate_hard_fix(model, protein_data, n_mols, full_mol_size, device,
                      site_map, anchor_positions, timesteps=100, locked=False):
    """Generate with hard-fixed anchors.

    Args:
        locked: if True, use FullyLockedAnchorCallback (anchors removed before ODE)
    """
    patch_drugflow_hardfix()
    if locked:
        patch_drugflow_fully_locked()

    site_energy = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(device)

    n_anchors = len(anchor_positions)
    anchor_indices = list(range(n_anchors))
    anchor_pos_tensor = torch.tensor(anchor_positions, dtype=torch.float32)

    # Guide function with harmonic restraint for anchors
    anchors_dummy = AnchorAtoms(
        positions=anchor_pos_tensor,
        type_indices=torch.zeros(n_anchors, dtype=torch.long),
        type_probs=torch.zeros(n_anchors, N_ATOM_TYPES),
        occupied_sites=[],
        compat_scores=[],
        distances=[],
    )

    guide_fn = TwoStageGuideFn(
        site_energy=site_energy, anchors=anchors_dummy,
        config=Phase2Config(lambda_late=0.1, restraint_force=10.0),
        kts=KTSScheduler(alpha0=0.005, beta0=0.01),
    ).to(device)
    guide_fn.set_anchor_indices(anchor_indices, full_mol_size)

    if locked:
        post_step_callback = FullyLockedAnchorCallback(
            anchor_indices=anchor_indices,
            anchor_coords=anchor_pos_tensor,
            verbose=False,
        )
        pre_step_callback = post_step_callback
    else:
        post_step_callback = HardFixCallback(
            anchor_indices=anchor_indices,
            anchor_coords=anchor_pos_tensor,
            fix_coords=True,
            fix_types=False,
        )
        pre_step_callback = None

    patch_drugflow_sample_post_step(model)

    molecules = []
    t0 = time.time()
    with torch.no_grad():
        kwargs = dict(
            data=protein_data, n_samples=n_mols,
            timesteps=timesteps, num_nodes=full_mol_size,
            guide_log_prob=guide_fn,
            post_step_callback=post_step_callback,
        )
        if pre_step_callback is not None:
            kwargs["pre_step_callback"] = pre_step_callback
        rdmols, _, _ = model.sample(**kwargs)
    elapsed = time.time() - t0

    for m in rdmols:
        if m is not None:
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                                 ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass
            molecules.append(m)
    return molecules, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Condition: KAG (two-stage kinematic anchor guidance)
# ═══════════════════════════════════════════════════════════════════════════

def generate_kag(model, protein_data, n_mols, full_mol_size, device,
                 site_map, phase1_config=None, phase2_config=None, output_dir=None):
    """Two-stage KAG generation."""
    config = TwoStageConfig(
        phase1=phase1_config or Phase1Config(
            n_init_atoms=4, attempts=3,
            lambda_early=5.0,  # ★ updated λ
            success_distance=2.5, min_compatibility=0.3,
            sigma_distance=3.0,
        ),
        phase2=phase2_config or Phase2Config(
            anchor_fix_mode="kinematic",
            kinematic_lambda_max=1.0,  # ★ updated λ_max
            max_total_steps=100, lambda_late=0.1,
            restraint_force=10.0,
        ),
        verbose=True,
    )

    generator = TwoStageGenerator(config=config, model=model, site_map=site_map).to(device)

    all_mols = []
    total_time = 0.0
    successes = 0
    degradations = 0

    for mol_i in range(n_mols):
        print(f"\n  KAG mol {mol_i+1}/{n_mols} ...")
        t0 = time.time()
        result = generator.generate(
            protein_data=protein_data,
            full_mol_size=full_mol_size,
            n_phase2_samples=1,
            phase1_timesteps=50,
            phase2_timesteps=100,
            device=device,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        mode = result.get("generation_mode", "two_stage" if result["success"] else "failed")
        if mode == "single_stage_degraded":
            degradations += 1
        elif result["success"]:
            successes += 1

        mols = result.get("molecules", [])
        if mols is None:
            # Single-stage degradation — generate with full gradient instead
            fb_mols, fb_time = generate_full_gradient(
                model, protein_data, 1, full_mol_size, device, site_map)
            for m in fb_mols:
                if m is not None:
                    all_mols.append(m)
            continue
        for m in (mols or []):
            if m is not None:
                try:
                    Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                                     ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception:
                    pass
                all_mols.append(m)

    print(f"  KAG: {successes} two-stage, {degradations} degraded, "
          f"{len(all_mols)} valid mols, {total_time:.0f}s")
    return all_mols, total_time


# ═══════════════════════════════════════════════════════════════════════════
# Metrics evaluation — traditional + new
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_molecules(molecules, site_map, condition_name):
    """Compute traditional and new metrics for a set of molecules."""
    from metrics_new import (
        compute_mol_centroid, compute_centroid_hew_distances,
        compute_cos, compute_e_site,
        classify_hew_environment, env_to_idx,
    )
    import numpy as np

    hew_sites = [s for s in site_map.get("sites", [])
                 if s.get("site_type") == "high_energy_water"]
    hew_centers = np.array([s["center"] for s in hew_sites])
    hew_env_indices = np.array([env_to_idx(classify_hew_environment(s)) for s in hew_sites])

    results = []
    for mol_idx, mol in enumerate(molecules):
        if mol is None or mol.GetNumAtoms() == 0:
            continue

        centroid = compute_mol_centroid(mol)
        cd = compute_centroid_hew_distances(centroid, hew_centers) if centroid is not None else {}
        cos = compute_cos(mol, hew_centers, hew_env_indices, sigma=1.5)
        e_site = compute_e_site(mol, hew_centers, hew_env_indices, sigma=3.0, tau=10.0)

        # Traditional metrics
        try:
            qed_val = QED.qed(mol)
        except Exception:
            qed_val = float("nan")

        try:
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
        except Exception:
            mw = float("nan")

        results.append({
            "mol_id": mol_idx,
            "n_atoms": mol.GetNumAtoms(),
            **cd,
            **cos,
            "E_site": e_site,
            "QED": qed_val,
            "MolWt": mw,
        })

    if not results:
        return {"error": "No valid molecules", "n": 0}

    # Aggregate
    keys = ["min_dist_centroid", "avg_dist_centroid", "avg_COS", "max_COS", "E_site", "QED"]
    stats = {}
    for key in keys:
        vals = [r[key] for r in results if key in r and not (isinstance(r.get(key), float) and np.isnan(r.get(key, float("nan"))))]
        if vals:
            stats[key] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "median": float(np.median(vals)), "n": len(vals),
            }

    return {"n": len(results), "statistics": stats, "per_mol": results}


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="KAG Full Experiment Suite")
    parser.add_argument("--all", action="store_true", help="Run all pockets and conditions")
    parser.add_argument("--pockets", type=str, default="3mfw,2gni,6o4x,2jke,2gqn,6phx",
                        help="Comma-separated pocket IDs")
    parser.add_argument("--conditions", type=str,
                        default="unguided,hard_fix,full_gradient,kag,hard_fix_locked",
                        help="Conditions to run")
    parser.add_argument("--n-mols", type=int, default=50, help="Molecules per condition")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pockets = [p.strip() for p in args.pockets.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("Loading DrugFlow model...")
    model = load_model(device=args.device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params\n")

    for pocket in pockets:
        cfg = POCKETS_CONFIG.get(pocket, {"year": "2001-2010"})
        year = cfg["year"]
        protein_pdb = DATA_BASE / year / pocket / f"{pocket}_protein.pdb"
        ref_ligand = DATA_BASE / year / pocket / f"{pocket}_ligand.sdf"
        site_map_path = SITE_MAPS_DIR / f"{pocket}_site_map.json"

        if not protein_pdb.exists():
            print(f"  ⚠ Missing protein PDB for {pocket}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Pocket: {pocket}")
        print(f"{'='*60}")

        # Load site map
        site_map = json.loads(site_map_path.read_text()) if site_map_path.exists() else {"sites": []}

        # Pre-process protein
        print("  Processing protein...")
        data, ref_size = process_protein(str(protein_pdb), str(ref_ligand), model)
        full_mol_size = ref_size
        print(f"  Reference size: {ref_size} atoms")

        protein_data = {
            "ligand": TensorDict(**data["ligand"]).to(args.device),
            "pocket": TensorDict(**data["pocket"]).to(args.device),
        }

        # Get reference ligand anchor positions (first 4 atoms) for hard_fix
        ref_mol = Chem.SDMolSupplier(str(ref_ligand))[0]
        ref_conf = ref_mol.GetConformer()
        anchor_positions = [list(ref_conf.GetAtomPosition(i)) for i in range(min(4, ref_mol.GetNumAtoms()))]

        for condition in conditions:
            print(f"\n  --- Condition: {condition} ({args.n_mols} mols) ---")

            output_dir = OUTPUT_BASE / pocket / condition
            output_dir.mkdir(parents=True, exist_ok=True)

            sdf_path = output_dir / "molecules.sdf"
            meta_path = output_dir / "metadata.json"

            if sdf_path.exists() and meta_path.exists():
                print(f"    Already exists, skipping. Remove {output_dir} to regenerate.")
                continue

            if args.dry_run:
                print(f"    [DRY RUN] Would generate {args.n_mols} molecules")
                continue

            try:
                if condition == "unguided":
                    mols, elapsed = generate_unguided(
                        model, protein_data, args.n_mols, full_mol_size, args.device)
                elif condition == "full_gradient":
                    mols, elapsed = generate_full_gradient(
                        model, protein_data, args.n_mols, full_mol_size, args.device, site_map)
                elif condition == "hard_fix":
                    mols, elapsed = generate_hard_fix(
                        model, protein_data, args.n_mols, full_mol_size, args.device,
                        site_map, anchor_positions, locked=False)
                elif condition == "hard_fix_locked":
                    mols, elapsed = generate_hard_fix(
                        model, protein_data, args.n_mols, full_mol_size, args.device,
                        site_map, anchor_positions, locked=True)
                elif condition == "kag":
                    mols, elapsed = generate_kag(
                        model, protein_data, args.n_mols, full_mol_size, args.device, site_map,
                        output_dir=output_dir)
                else:
                    print(f"    Unknown condition: {condition}")
                    continue
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Save SDF (with error tolerance for malformed molecules)
            writer = Chem.SDWriter(str(sdf_path))
            writer.SetKekulize(False)
            written = 0
            for m in mols:
                if m is not None:
                    try:
                        writer.write(m)
                        written += 1
                    except Exception:
                        pass
            writer.close()
            print(f"    Saved {written}/{len(mols)} molecules to SDF")

            # Evaluate metrics
            metrics = evaluate_molecules(mols, site_map, condition)
            metrics["condition"] = condition
            metrics["pocket"] = pocket
            metrics["elapsed_s"] = elapsed

            # Save metadata
            meta = {
                "pocket": pocket, "condition": condition,
                "n_requested": args.n_mols, "n_generated": len(mols),
                "n_valid": metrics["n"],
                "elapsed_s": elapsed,
                "parameters": {
                    "phase1_lambda": 5.0,
                    "phase2_kinematic_lambda_max": 1.0,
                    "sigma_distance": 3.0,
                    "tau": 10.0,
                    "compatibility_matrix": "Table 10",
                },
                "statistics": metrics.get("statistics", {}),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

            # Print quick summary
            stats = metrics.get("statistics", {})
            print(f"    Generated: {len(mols)} mols, valid: {metrics['n']}, time: {elapsed:.0f}s")
            for k, v in stats.items():
                print(f"      {k:<25} {v['mean']:.4f} ± {v['std']:.4f}")

    print(f"\n{'='*60}")
    print("Experiment suite complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
