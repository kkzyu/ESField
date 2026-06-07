#!/usr/bin/env python3
"""v7.1 — Hard-Fix Two-Stage Generation on Actionable Pockets (2gni, 3mfw, 6o4x).

Key improvement over v7.0: uses HARD COORDINATE OVERWRITE instead of harmonic
restraint for anchor atom preservation in Phase 2.

Run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v71_actionable.py [--pockets 2gni,3mfw,6o4x]
"""

import json, os, sys, time, warnings, argparse
from pathlib import Path

# ── DrugFlow imports (must come first) ──
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

# ── ESField imports ──
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
    _tensors_from_rdmol, AnchorAtoms,
)
from guidance.hard_fix import patch_drugflow_hardfix, HardFixCallback
from evaluation.site_occupancy import site_occupancy_summary, direct_occupancy_rate
from evaluation.posu import compute_posu

# ── Apply hardfix patch ──
patch_drugflow_hardfix()

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
PDB_ROOT = "/root/autodl-tmp/data/PDB/P-L"
SITE_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/site_maps"
OUTPUT_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_actionable"
DEVICE = "cuda:0"

# ── Config ──
PHASE1_LAMBDA = 5.0
PHASE1_STEPS = 100
PHASE1_ATOMS = 4
PHASE1_ATTEMPTS = 3
PHASE1_PER_ATTEMPT = 5
SUCCESS_DISTANCE = 2.5
MIN_COMPATIBILITY = -0.5  # Accept any atom near site (v6-D.2 baseline = 0 occ)

PHASE2_STEPS = 100
N_PHASE2_SAMPLES = 20


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


def check_anchor_quality(anchors, energy_fn, site_map):
    """Validate anchor atoms before Phase 2.

    Returns (passed, issues) where issues is a list of warning strings.
    """
    issues = []
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]

    for i in range(anchors.n_anchors):
        d = anchors.distances[i]
        compat = anchors.compat_scores[i]

        if d > 2.5:
            issues.append(f"Anchor {i}: d={d:.2f}Å > 2.5Å (too far from HEW site)")
        elif d > 2.0:
            issues.append(f"Anchor {i}: d={d:.2f}Å > 2.0Å (marginal, prefer tighter)")

        if compat < 0.5:
            issues.append(f"Anchor {i}: compat={compat:.3f} < 0.5 (low compatibility)")
        elif compat < 0.3:
            issues.append(f"Anchor {i}: compat={compat:.3f} < 0.3 (very low compatibility)")

    passed = len([i for i in issues if "> 2.5" in i or "< 0.3" in i]) == 0
    return passed, issues


def run_phase1(model, protein_data, energy_fn, verbose=True):
    """Phase 1: generate small fragments that occupy HEW sites."""
    kts = KTSScheduler(alpha0=0.01, beta0=0.01)
    guide_fn = _Phase1GuideFn(
        site_energy=energy_fn, lambda_guide=PHASE1_LAMBDA,
        guidance_start=0.05, guidance_end=0.95,
        grad_clip=1.0, kts=kts,
    ).to(DEVICE)

    best_d_overall = float("inf")

    for attempt in range(PHASE1_ATTEMPTS):
        t0 = time.time()
        with torch.no_grad():
            rdmols, _, _ = model.sample(
                protein_data, n_samples=PHASE1_PER_ATTEMPT,
                timesteps=PHASE1_STEPS, num_nodes=PHASE1_ATOMS,
                guide_log_prob=guide_fn,
            )
        elapsed = time.time() - t0

        for idx, mol in enumerate(rdmols):
            if mol is None:
                continue
            x, h = _tensors_from_rdmol(mol, device=DEVICE)
            if x is None:
                continue

            diag = _compute_diagnostics(
                x, h, energy_fn, SUCCESS_DISTANCE, MIN_COMPATIBILITY
            )

            if diag["best_distance"] < best_d_overall:
                best_d_overall = diag["best_distance"]

            if diag["success"]:
                cfg = Phase1Config(
                    success_distance=SUCCESS_DISTANCE,
                    min_compatibility=MIN_COMPATIBILITY,
                    anchor_selection="best_per_site",
                )
                anchors = _extract_anchors(x, h, energy_fn, diag, cfg)
                if anchors is not None and anchors.n_anchors > 0:
                    if verbose:
                        types_str = ",".join(
                            f"{ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]}"
                            for i in range(anchors.n_anchors)
                        )
                        print(f"  ✓ Phase1 success (attempt {attempt+1}, mol {idx}): "
                              f"{anchors.n_anchors} anchor(s) [{types_str}], "
                              f"best_d={diag['best_distance']:.2f}Å, "
                              f"time={elapsed:.1f}s")
                    return anchors, diag

        if verbose:
            print(f"  Attempt {attempt+1}: {PHASE1_PER_ATTEMPT} fragments, "
                  f"best_d={best_d_overall:.2f}Å, elapsed={elapsed:.1f}s")

    return None, None


def run_phase2(model, protein_data, energy_fn, anchors, ref_size, verbose=True):
    """Phase 2: generate full molecules with hard anchor fix.

    Replicates DrugFlow's sample() logic but with post_step_callback for
    hard coordinate overwrite.
    """
    n_samples = N_PHASE2_SAMPLES
    timesteps = PHASE2_STEPS

    # Build hard fix callback
    anchor_indices = list(range(anchors.n_anchors))
    anchor_coords = anchors.positions.clone()
    anchor_h = anchors.type_probs.clone() if anchors.type_probs is not None else None

    callback = HardFixCallback(
        anchor_indices=anchor_indices,
        anchor_coords=anchor_coords,
        anchor_h=None,  # DrugFlow uses different type dims (14 vs 11)
        fix_coords=True,
        fix_types=False,  # Only fix coordinates; types handled by model
        verbose=False,
    )

    # Weak site guidance for remaining atoms
    kts = KTSScheduler(alpha0=0.005, beta0=0.01)
    from guidance.two_stage_generation import TwoStageGuideFn, Phase2Config
    cfg = Phase2Config(
        fix_atoms=True, restraint_force=0.0,  # No harmonic restraint (hard fix instead)
        lambda_late=0.1, guidance_start=0.1, guidance_end=0.90,
        grad_clip=0.3,
    )
    guide_fn = TwoStageGuideFn(energy_fn, anchors, cfg, kts).to(DEVICE)
    guide_fn.set_anchor_indices(anchor_indices, ref_size)

    # ── Replicate DrugFlow sample() setup ──
    from src.data import data_utils
    from src.data.molecule_builder import build_molecule
    from src import utils
    from itertools import accumulate

    # Repeat for batch
    data = protein_data
    if len(data['pocket']['x']) > 0:
        pocket = data_utils.repeat_items(data['pocket'], n_samples)
    else:
        pocket = data_utils.Residues(**{key: value for key, value in data['pocket'].items()})
        pocket['name'] = pocket['name'] * n_samples
        pocket['size'] = pocket['size'].repeat(n_samples)
        pocket['n_bonds'] = pocket['n_bonds'].repeat(n_samples)

    _ligand = data_utils.repeat_items(data['ligand'], n_samples)

    # Init from prior
    num_nodes = model.parse_num_nodes_spec(
        {"ligand": _ligand, "pocket": pocket},
        spec=ref_size, size_model=None
    )
    if pocket['x'].numel() > 0:
        ligand = model.init_ligand(num_nodes, pocket)
    else:
        ligand = model.init_ligand(num_nodes, _ligand)
    pocket = model.init_pocket(pocket)

    # ── Run simulate with hard fix ──
    t0 = time.time()
    with torch.no_grad():
        out_tensors_ligand, out_tensors_pocket = model.simulate(
            ligand, pocket, timesteps, 0.0, 1.0,
            guide_log_prob=guide_fn,
            post_step_callback=callback,
        )
    elapsed = time.time() - t0

    # ── Post-process: convert to RDKit molecules (replicating sample() logic) ──
    x = out_tensors_ligand['x'].detach().cpu()
    ligand_type = out_tensors_ligand['h'].argmax(1).detach().cpu()
    edge_type = out_tensors_ligand['e'].argmax(1).detach().cpu()
    lig_mask = ligand['mask'].detach().cpu()
    lig_bonds = ligand['bonds'].detach().cpu()
    lig_edge_mask = ligand['edge_mask'].detach().cpu()
    sizes = torch.unique(ligand['mask'], return_counts=True)[1].tolist()
    offsets = list(accumulate(sizes[:-1], initial=0))

    mol_kwargs = {
        'coords': utils.batch_to_list(x, lig_mask),
        'atom_types': utils.batch_to_list(ligand_type, lig_mask),
        'bonds': utils.batch_to_list_for_indices(lig_bonds, lig_edge_mask, offsets),
        'bond_types': utils.batch_to_list(edge_type, lig_edge_mask)
    }
    mol_kwargs = [{k: v[i] for k, v in mol_kwargs.items()}
                  for i in range(len(mol_kwargs['coords']))]

    rdmols = [build_molecule(
        **m, atom_decoder=model.atom_decoder, bond_decoder=model.bond_decoder)
        for m in mol_kwargs
    ]

    valid = [m for m in rdmols if m is not None]
    if verbose:
        print(f"  Phase2: {len(valid)}/{n_samples} valid, {elapsed:.1f}s, "
              f"hardfix calls={callback.n_calls}")

    return valid, callback.n_calls


def evaluate_molecules(valid_mols, site_map, pocket_name):
    """Compute all metrics for a set of generated molecules."""
    # Site occupancy
    occ = site_occupancy_summary(valid_mols, site_map, threshold=2.5)
    dor = occ["direct_occupancy"]
    bcd = occ["compatible_distance"]

    # Quality
    qeds, mws, logps = [], [], []
    for m in valid_mols:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            qeds.append(QED.qed(m))
            mws.append(Descriptors.MolWt(m))
            logps.append(Descriptors.MolLogP(m))
        except Exception:
            pass

    # POSU
    posu_vals, hewu_vals = [], []
    for m in valid_mols:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            p = compute_posu(m, site_map)
            posu_vals.append(p["posu"])
            hewu_vals.append(p["hew_mean"])
        except Exception:
            pass

    return {
        "pocket": pocket_name,
        "n_valid": len(valid_mols),
        "n_total": len(valid_mols),  # will be overridden by caller
        "direct_occ_rate": dor["rate"],
        "n_occupied": dor["n_occupied"],
        "best_compat_d_mean": bcd["mean"],
        "best_compat_d_min": bcd["min"],
        "n_sites_occupied": bcd["n_sites_occupied"],
        "n_sites_total": bcd["n_sites_total"],
        "qed_mean": float(np.mean(qeds)) if qeds else 0,
        "qed_std": float(np.std(qeds)) if qeds else 0,
        "mw_mean": float(np.mean(mws)) if mws else 0,
        "logp_mean": float(np.mean(logps)) if logps else 0,
        "posu_mean": float(np.mean(posu_vals)) if posu_vals else 0,
        "posu_std": float(np.std(posu_vals)) if posu_vals else 0,
        "hewu_mean": float(np.mean(hewu_vals)) if hewu_vals else 0,
    }


def run_pocket(pocket_name, model, verbose=True):
    """Run complete v7.1 pipeline on one pocket."""
    print(f"\n{'='*70}")
    print(f"  POCKET: {pocket_name}")
    print(f"{'='*70}")

    # Load site map
    site_map_path = os.path.join(SITE_DIR, f"{pocket_name}_site_map.json")
    site_map = json.load(open(site_map_path))
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if verbose:
        envs = {}
        for s in hew_sites:
            e = classify_hew_environment(s)
            envs[e] = envs.get(e, 0) + 1
        print(f"  HEW sites: {len(hew_sites)} ({envs})")

    # Build site energy
    energy_fn = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(DEVICE)

    # Find protein/ligand
    import glob
    pdirs = glob.glob(os.path.join(PDB_ROOT, "*", pocket_name))
    if not pdirs:
        print(f"  ERROR: protein dir not found for {pocket_name}")
        return None
    pdir = pdirs[0]
    protein_pdb = os.path.join(pdir, f"{pocket_name}_protein.pdb")
    ligand_sdf = os.path.join(pdir, f"{pocket_name}_ligand.sdf")

    # Process protein
    data, ref_size = process_protein(protein_pdb, ligand_sdf, model)
    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(DEVICE),
        "pocket": TensorDict(**data["pocket"]).to(DEVICE),
    }
    if verbose:
        print(f"  Reference ligand: {ref_size} atoms, "
              f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Phase 1
    if verbose:
        print(f"  Phase 1 (λ={PHASE1_LAMBDA}, {PHASE1_ATOMS} atoms, "
              f"{PHASE1_STEPS} steps)...")
    anchors, phase1_diag = run_phase1(model, protein_data, energy_fn, verbose)

    if anchors is None:
        print(f"  ✗ Phase 1 FAILED — no anchors found")
        return {"pocket": pocket_name, "phase1_success": False, "phase2_ran": False}

    # Check anchor quality
    passed, issues = check_anchor_quality(anchors, energy_fn, site_map)
    if verbose:
        for issue in issues:
            print(f"  [Quality] {issue}")
        print(f"  Anchor quality check: {'PASSED' if passed else 'WARNINGS'}")

    # Save anchors
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anchor_path = os.path.join(OUTPUT_DIR, f"{pocket_name}_anchors.json")
    json.dump(anchors.to_dict(), open(anchor_path, "w"), indent=2)

    # Phase 2
    if verbose:
        print(f"  Phase 2 ({N_PHASE2_SAMPLES} samples, {PHASE2_STEPS} steps, "
              f"hard fix)...")
    phase2_mols, n_hardfix_calls = run_phase2(
        model, protein_data, energy_fn, anchors, ref_size, verbose
    )

    if not phase2_mols:
        print(f"  ✗ Phase 2 produced no valid molecules")
        return {"pocket": pocket_name, "phase1_success": True, "phase2_ran": True,
                "n_valid": 0}

    # Save molecules
    sdf_path = os.path.join(OUTPUT_DIR, f"{pocket_name}_molecules.sdf")
    writer = Chem.SDWriter(sdf_path)
    writer.SetKekulize(False)
    for m in phase2_mols:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            pass
        writer.write(m)
    writer.close()

    # Evaluate
    metrics = evaluate_molecules(phase2_mols, site_map, pocket_name)
    metrics["n_total"] = N_PHASE2_SAMPLES
    metrics["phase1_success"] = True
    metrics["phase2_ran"] = True
    metrics["n_anchors"] = anchors.n_anchors
    metrics["n_hardfix_calls"] = n_hardfix_calls
    metrics["anchor_distances"] = anchors.distances
    metrics["anchor_compat_scores"] = anchors.compat_scores

    if verbose:
        print(f"\n  ── Results for {pocket_name} ──")
        print(f"  Valid: {metrics['n_valid']}/{metrics['n_total']}")
        print(f"  DirectOcc: {metrics['direct_occ_rate']:.3f} "
              f"({metrics['n_occupied']}/{metrics['n_valid']})")
        print(f"  BestCompatD: mean={metrics['best_compat_d_mean']:.2f}Å, "
              f"min={metrics['best_compat_d_min']:.2f}Å")
        print(f"  Sites occupied: {metrics['n_sites_occupied']}/{metrics['n_sites_total']}")
        print(f"  QED: {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}")
        print(f"  POSU: {metrics['posu_mean']:.3f}, HEWU: {metrics['hewu_mean']:.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets", default="2gni,3mfw,6o4x",
                        help="Comma-separated pocket names")
    parser.add_argument("--phase1-lambda", type=float, default=PHASE1_LAMBDA)
    parser.add_argument("--n-samples", type=int, default=N_PHASE2_SAMPLES)
    args = parser.parse_args()

    pockets = [p.strip() for p in args.pockets.split(",")]

    print("=" * 70)
    print("v7.1 — Hard-Fix Two-Stage on Actionable Pockets")
    print(f"Pockets: {pockets}")
    print(f"Phase1: λ={args.phase1_lambda}, {PHASE1_ATOMS} atoms, "
          f"{PHASE1_STEPS} steps, {PHASE1_ATTEMPTS} attempts")
    print(f"Phase2: {args.n_samples} samples, {PHASE2_STEPS} steps, HARD FIX")
    print("=" * 70)

    # Load model once
    print("\n[0] Loading DrugFlow...")
    model = load_model(DRUGFLOW_CKPT, device=DEVICE)
    print(f"    GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    all_metrics = {}
    total_start = time.time()

    for pocket in pockets:
        metrics = run_pocket(pocket, model, verbose=True)
        if metrics:
            all_metrics[pocket] = metrics

    total_elapsed = time.time() - total_start

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY — v7.1 Actionable Pockets")
    print(f"{'='*70}")
    print(f"{'Pocket':<8} {'Ph1':>4} {'Valid':>6} {'DirectOcc':>10} "
          f"{'BestD':>7} {'SitesOcc':>9} {'QED':>7} {'POSU':>7} {'HEWU':>7}")
    print("-" * 70)

    for pocket in pockets:
        m = all_metrics.get(pocket, {})
        if not m:
            print(f"{pocket:<8} {'FAIL':>4}")
            continue
        ph1 = "OK" if m.get("phase1_success") else "FAIL"
        print(f"{pocket:<8} {ph1:>4} {m.get('n_valid',0):>4}/{m.get('n_total','?'):>2} "
              f"{m.get('direct_occ_rate',0):>9.3f} "
              f"{m.get('best_compat_d_min','N/A'):>6} "
              f"{m.get('n_sites_occupied',0):>4}/{m.get('n_sites_total',0):>4} "
              f"{m.get('qed_mean',0):>6.3f} "
              f"{m.get('posu_mean',0):>6.3f} "
              f"{m.get('hewu_mean',0):>6.3f}")

    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Save all metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "v71_all_metrics.json")
    json.dump(all_metrics, open(metrics_path, "w"), indent=2, default=str)
    print(f"Metrics saved to {metrics_path}")

    return all_metrics


if __name__ == "__main__":
    main()
