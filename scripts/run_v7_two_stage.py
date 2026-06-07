#!/usr/bin/env python3
"""v7 — Two-Stage Hierarchical Latent Guidance for DrugFlow.

Phase 1 (OCCUPY):  Generate a small fragment (3-5 atoms) with strong
    site-compatibility guidance to place atoms into candidate HEW sites.

Phase 2 (CONNECT): Grow a full drug-like molecule around the anchor
    atoms, with harmonic restraints preserving the occupied positions.

This script implements the v7 approach that addresses the NO-GO conclusion
from v6-D.2: instead of only nudging coordinates, v7 explicitly grows
fragments at target sites (topology control) and then connects them.

Usage:
    PYTHONPATH=src python scripts/run_v7_two_stage.py \
        --protein-pdb <pdb> \
        --ref-ligand <sdf> \
        --site-map <site_map.json> \
        --output-dir <dir> \
        --full-mol-size 25 \
        [--n-phase2-samples 5]

Dependencies:
    pip install rdkit pyyaml
    (OpenBabel for post-processing: conda install -c conda-forge openbabel)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# v7 imports
from guidance.latent_guidance import (
    SiteCompatibilityEnergy,
    build_site_energy_from_map,
    classify_hew_environment,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    TwoStageGenerator,
    TwoStageConfig,
    Phase1Config,
    Phase2Config,
    AnchorAtoms,
    TwoStageGuideFn,
)
from evaluation.site_occupancy import (
    evaluate_sdf_occupancy,
    site_occupancy_summary,
)
from evaluation.posu import compute_posu

# DrugFlow integration (reuse existing infrastructure)
from scripts.drugflow_esfield_guide import (
    load_drugflow_model,
    process_protein_for_drugflow,
)

DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"
DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="v7 Two-Stage Hierarchical Latent Guidance for DrugFlow"
    )
    # Required inputs
    parser.add_argument("--protein-pdb", required=True, help="Protein PDB file")
    parser.add_argument("--ref-ligand", required=True, help="Reference ligand SDF")
    parser.add_argument("--site-map", required=True, help="Site map JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    # Generation parameters
    parser.add_argument("--full-mol-size", type=int, default=25,
                        help="Number of heavy atoms in the full molecule")
    parser.add_argument("--phase1-atoms", type=int, default=4,
                        help="Number of atoms in Phase 1 fragment")
    parser.add_argument("--n-phase2-samples", type=int, default=5,
                        help="Number of full molecules to generate in Phase 2")
    parser.add_argument("--phase1-attempts", type=int, default=3,
                        help="Max Phase 1 retry attempts")
    parser.add_argument("--phase1-timesteps", type=int, default=50,
                        help="ODE steps for Phase 1")
    parser.add_argument("--phase2-timesteps", type=int, default=100,
                        help="ODE steps for Phase 2")

    # Guidance
    parser.add_argument("--lambda-early", type=float, default=0.5,
                        help="Guidance strength for Phase 1")
    parser.add_argument("--lambda-late", type=float, default=0.1,
                        help="Guidance strength for Phase 2")
    parser.add_argument("--restraint-force", type=float, default=10.0,
                        help="Harmonic restraint force constant for anchors")
    parser.add_argument("--anchor-fix-mode", type=str, default="hard",
                        choices=["hard", "annealing", "soft", "kinematic"],
                        help="Anchor fixation strategy (★ 'kinematic' = SV-Flow fusion)")

    # Kinematic guidance (SV-Flow fusion)
    parser.add_argument("--kinematic-lambda-max", type=float, default=0.5,
                        help="λ_max for kinematic anchor guidance")
    parser.add_argument("--kinematic-profile", type=str, default="quadratic",
                        choices=["quadratic", "constant", "late_onset", "linear"],
                        help="λ(t) decay profile for kinematic guidance")
    parser.add_argument("--kinematic-grad-clip", type=float, default=0.5,
                        help="Max per-step CoM correction (Å)")
    parser.add_argument("--no-kpe-tracking", action="store_true",
                        help="Disable KPE diagnostic tracking")

    # KTS
    parser.add_argument("--kts-alpha0", type=float, default=0.01,
                        help="KTS early boost strength")
    parser.add_argument("--kts-beta0", type=float, default=0.01,
                        help="KTS late damping strength")
    parser.add_argument("--kts-tau-split", type=float, default=0.6,
                        help="KTS transition point")
    parser.add_argument("--kts-k", type=float, default=3.0,
                        help="KTS exponential stiffness")

    # Site filtering
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum site confidence for HEW")
    parser.add_argument("--success-distance", type=float, default=2.5,
                        help="Distance threshold for occupancy (Å)")
    parser.add_argument("--min-compatibility", type=float, default=0.5,
                        help="Minimum compatibility for successful occupation")

    # Model
    parser.add_argument("--drugflow-ckpt", default=DRUGFLOW_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # Post-processing
    parser.add_argument("--no-minimize", action="store_true",
                        help="Skip force-field minimization")
    parser.add_argument("--force-field", default="MMFF94",
                        choices=["MMFF94", "UFF"])

    # Misc
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # ------------------------------------------------------------------
    # 1. Load DrugFlow model
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("v7 Two-Stage Hierarchical Latent Guidance")
        print("=" * 70)
        print(f"\nLoading DrugFlow from {args.drugflow_ckpt}")

    model = load_drugflow_model(args.drugflow_ckpt, device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model loaded: {n_params:,} parameters")

    # ------------------------------------------------------------------
    # 2. Load site map and preprocess
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nLoading site map from {args.site_map}")

    site_map = json.loads(Path(args.site_map).read_text())
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if verbose:
        print(f"  HEW candidate sites: {len(hew_sites)}")
        for s in hew_sites[:5]:
            env = classify_hew_environment(s)
            conf = s.get("confidence", 1.0)
            center = s["center"]
            print(f"    site {s.get('site_id', '?'):>4}  env={env:<20}  "
                  f"conf={conf:.2f}  center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        if len(hew_sites) > 5:
            print(f"    ... and {len(hew_sites) - 5} more")

    if not hew_sites:
        print("  WARNING: No HEW sites found. Two-stage generation is not applicable.")
        print("  Falling back to unconditional generation (Phase 2 only).")

    # ------------------------------------------------------------------
    # 3. Process protein for DrugFlow
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nProcessing protein {args.protein_pdb}")

    data, ref_size = process_protein_for_drugflow(
        args.protein_pdb, args.ref_ligand, model
    )
    full_mol_size = args.full_mol_size or ref_size

    if verbose:
        print(f"  Reference ligand: {ref_size} atoms")
        print(f"  Full molecule target: {full_mol_size} atoms")

    # Move data to device
    from src.data.data_utils import TensorDict
    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(args.device),
        "pocket": TensorDict(**data["pocket"]).to(args.device),
    }

    # ------------------------------------------------------------------
    # 4. Build TwoStageGenerator
    # ------------------------------------------------------------------
    config = TwoStageConfig(
        phase1=Phase1Config(
            n_init_atoms=args.phase1_atoms,
            attempts=args.phase1_attempts,
            success_distance=args.success_distance,
            min_compatibility=args.min_compatibility,
            lambda_early=args.lambda_early,
            guidance_start=0.05,
            guidance_end=0.95,
            grad_clip=0.5,
            sigma_distance=3.0,
            kts_alpha0=args.kts_alpha0,
            kts_beta0=args.kts_beta0,
        ),
        phase2=Phase2Config(
            anchor_fix_mode=args.anchor_fix_mode,
            fix_atoms=True,
            restraint_force=args.restraint_force,
            max_total_steps=args.phase2_timesteps,
            lambda_late=args.lambda_late,
            guidance_start=0.1,
            guidance_end=0.90,
            grad_clip=0.3,
            kts_alpha0=args.kts_alpha0 * 0.5,
            kts_beta0=args.kts_beta0,
            # Kinematic (SV-Flow fusion)
            kinematic_lambda_max=args.kinematic_lambda_max,
            kinematic_profile=args.kinematic_profile,
            kinematic_grad_clip=args.kinematic_grad_clip,
            kinematic_track_kpe=not args.no_kpe_tracking,
        ),
        minimize=not args.no_minimize,
        force_field=args.force_field,
        verbose=verbose,
    )

    generator = TwoStageGenerator(
        config=config,
        model=model,
        site_map=site_map,
    ).to(args.device)

    # ------------------------------------------------------------------
    # 5. Run two-stage generation
    # ------------------------------------------------------------------
    t_start = time.time()

    if hew_sites:
        result = generator.generate(
            protein_data=protein_data,
            full_mol_size=full_mol_size,
            n_phase2_samples=args.n_phase2_samples,
            phase1_timesteps=args.phase1_timesteps,
            phase2_timesteps=args.phase2_timesteps,
            device=args.device,
        )
    else:
        # No HEW sites: skip Phase 1, run Phase 2 only (unconditional)
        if verbose:
            print("\n  No HEW sites — running unconditional generation only.")
        dummy_anchors = AnchorAtoms(
            positions=torch.zeros(0, 3),
            type_indices=torch.zeros(0, dtype=torch.long),
            type_probs=torch.zeros(0, N_ATOM_TYPES := 11),
            occupied_sites=[],
            compat_scores=[],
            distances=[],
        )
        molecules = generator.phase2_connect(
            protein_data=protein_data,
            anchors=dummy_anchors,
            full_mol_size=full_mol_size,
            n_samples=args.n_phase2_samples,
            timesteps=args.phase2_timesteps,
            device=args.device,
            anchor_atom_indices=[],
        )
        result = {
            "success": True,
            "anchors": dummy_anchors,
            "molecules": molecules,
            "phase1_log": [],
            "phase2_log": generator._phase2_log,
        }

    total_elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*70}")
        print(f"Generation complete ({total_elapsed:.1f}s total)")

    # Save molecules
    molecules = result["molecules"]
    if molecules:
        sdf_path = str(output_dir / "generated_molecules.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.SetKekulize(False)
        valid_count = 0
        for m in molecules:
            if m is not None:
                try:
                    Chem.SanitizeMol(
                        m,
                        Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    )
                except Exception:
                    pass
                writer.write(m)
                valid_count += 1
        writer.close()
        if verbose:
            print(f"  Saved {valid_count} molecules to {sdf_path}")

        # Post-process: optional force-field minimization
        if not args.no_minimize:
            try:
                minimized_path = str(output_dir / "generated_molecules_minimized.sdf")
                _minimize_sdf(sdf_path, minimized_path, args.force_field, verbose)
            except Exception as e:
                if verbose:
                    print(f"  Minimization skipped: {e}")
    else:
        if verbose:
            print("  No molecules generated.")
        sdf_path = None

    # Save anchor details
    if result["anchors"] is not None and result["anchors"].n_anchors > 0:
        anchor_path = output_dir / "anchor_details.json"
        anchor_path.write_text(
            json.dumps(result["anchors"].to_dict(), indent=2),
            encoding="utf-8",
        )
        if verbose:
            print(f"  Saved anchor details to {anchor_path}")

    # Save generation logs
    logs = generator.get_logs()
    log_path = output_dir / "generation_log.json"
    log_path.write_text(json.dumps(logs, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"  Saved generation logs to {log_path}")

    # ------------------------------------------------------------------
    # 7. Compute evaluation metrics
    # ------------------------------------------------------------------
    if sdf_path and Path(sdf_path).exists():
        if verbose:
            print(f"\n{'='*70}")
            print("Computing site occupancy metrics...")

        occ_metrics = evaluate_sdf_occupancy(sdf_path, args.site_map)
        if verbose:
            dor = occ_metrics["direct_occupancy"]
            bcd = occ_metrics["compatible_distance"]
            print(f"  Direct occupancy rate: {dor['rate']:.3f} "
                  f"({dor['n_occupied']}/{dor['n_total']})")
            print(f"  Mean best compat distance: {bcd['mean']:.2f} Å")
            print(f"  Min best compat distance: {bcd['min']:.2f} Å")
            print(f"  Sites occupied (d≤2.5Å): {bcd['n_sites_occupied']}/{bcd['n_sites_total']}")

        # Save occupancy metrics
        occ_path = output_dir / "site_occupancy_metrics.json"
        occ_path.write_text(json.dumps(occ_metrics, indent=2, default=str),
                            encoding="utf-8")

        # Also compute POSU-v2.1 / HEWU for comparison with existing baselines
        if verbose:
            print(f"\n  Computing POSU-v2.1 / HEWU metrics...")
        posu_results = []
        for mol_path in [sdf_path]:
            mols = list(Chem.SDMolSupplier(mol_path, sanitize=False))
            for i, mol in enumerate(mols):
                if mol is None:
                    continue
                try:
                    Chem.SanitizeMol(
                        mol,
                        Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    )
                    posu = compute_posu(mol, site_map)
                    posu_results.append(posu)
                except Exception:
                    pass

        if posu_results:
            posu_vals = [p["posu"] for p in posu_results]
            hewu_vals = [p["hew_mean"] for p in posu_results]
            if verbose:
                print(f"    POSU-v2.1: {np.mean(posu_vals):.3f} ± {np.std(posu_vals):.3f}")
                print(f"    HEWU:      {np.mean(hewu_vals):.3f} ± {np.std(hewu_vals):.3f}")

            posu_path = output_dir / "posu_metrics.json"
            posu_path.write_text(
                json.dumps({
                    "posu_mean": float(np.mean(posu_vals)),
                    "posu_std": float(np.std(posu_vals)),
                    "hewu_mean": float(np.mean(hewu_vals)),
                    "hewu_std": float(np.std(hewu_vals)),
                    "per_mol": [
                        {"posu": p["posu"], "hewu": p["hew_mean"]}
                        for p in posu_results
                    ],
                }, indent=2),
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*70}")
        print("v7 Two-Stage Generation Summary")
        print(f"{'='*70}")
        print(f"  Phase 1 success: {result['success']}")
        if result["anchors"] is not None:
            print(f"  Anchor atoms:     {result['anchors'].n_anchors}")
        print(f"  Total time:       {total_elapsed:.1f}s")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*70}")

    # Write run metadata
    meta = {
        "version": "v7_two_stage",
        "protein_pdb": str(args.protein_pdb),
        "ref_ligand": str(args.ref_ligand),
        "site_map": str(args.site_map),
        "full_mol_size": full_mol_size,
        "phase1_atoms": args.phase1_atoms,
        "n_phase2_samples": args.n_phase2_samples,
        "lambda_early": args.lambda_early,
        "lambda_late": args.lambda_late,
        "phase1_success": result["success"],
        "n_anchors": result["anchors"].n_anchors if result["anchors"] else 0,
        "total_time": total_elapsed,
        "seed": args.seed,
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Post-processing: force-field minimization via OpenBabel
# ---------------------------------------------------------------------------


def _minimize_sdf(
    input_sdf: str,
    output_sdf: str,
    force_field: str = "MMFF94",
    verbose: bool = True,
) -> None:
    """Run force-field energy minimization on an SDF file using OpenBabel.

    Args:
        input_sdf: path to input SDF
        output_sdf: path to minimized output SDF
        force_field: "MMFF94" or "UFF"
        verbose: print progress
    """
    try:
        from openbabel import openbabel as ob
    except ImportError:
        if verbose:
            print("  [WARN] OpenBabel not available — skipping minimization.")
            print("    Install: conda install -c conda-forge openbabel")
        return

    conv = ob.OBConversion()
    conv.SetInAndOutFormats("sdf", "sdf")

    mol = ob.OBMol()
    if not conv.ReadFile(mol, input_sdf):
        if verbose:
            print("  [WARN] Could not read SDF for minimization.")
        return

    # Setup force field
    ff = ob.OBForceField.FindForceField(force_field)
    if ff is None:
        if verbose:
            print(f"  [WARN] Force field '{force_field}' not found — trying UFF.")
        ff = ob.OBForceField.FindForceField("UFF")
    if ff is None:
        if verbose:
            print("  [WARN] No force field available — skipping minimization.")
        return

    ff.Setup(mol)
    ff.ConjugateGradients(500, 1.0e-6)  # 500 steps, tight convergence
    ff.GetCoordinates(mol)

    conv.WriteFile(mol, output_sdf)
    if verbose:
        energy = ff.Energy()
        print(f"  Minimized: final energy = {energy:.1f} kcal/mol → {output_sdf}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
