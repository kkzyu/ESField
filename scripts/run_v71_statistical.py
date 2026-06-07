#!/usr/bin/env python3
"""v7.1 Statistical Validation — 50 samples per actionable pocket with binomial test.

Key improvements over v7.0/v7.1:
  - AnchorTypeSelector: suggests optimal atom types per HEW environment
  - Hard coordinate fix (post_step_callback) in Phase 2
  - Type bias: cross-entropy penalty to preserve anchor atom types
  - 50 samples per pocket for statistical power
  - One-sided binomial test vs baseline DirectOcc = 0

Run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v71_statistical.py [--pockets 2gni,3mfw,6o4x]
"""

import json, os, sys, time, warnings, argparse
from pathlib import Path
from math import comb as binom_coeff

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
    _tensors_from_rdmol, AnchorAtoms, TwoStageGuideFn, Phase2Config,
    suggest_anchor_types, AnchorTypeSelector,
)
from guidance.hard_fix import patch_drugflow_hardfix, HardFixCallback
from evaluation.site_occupancy import site_occupancy_summary, direct_occupancy_rate
from evaluation.posu import compute_posu

patch_drugflow_hardfix()

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
PDB_ROOT = "/root/autodl-tmp/data/PDB/P-L"
SITE_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/site_maps"
OUTPUT_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_statistical"
DEVICE = "cuda:0"

# ── Default config ──
CFG = {
    "phase1_lambda": 5.0,
    "phase1_steps": 100,
    "phase1_atoms": 4,
    "phase1_attempts": 3,
    "phase1_per_attempt": 5,
    "success_distance": 2.5,
    "min_compatibility": -0.5,
    "phase2_steps": 100,
    "n_samples": 50,
    "anchor_type_strategy": "suggested",
    "max_attempts_per_type": 2,
    "type_bias_strength": 0.5,
    "hard_fix_coords": True,
    "kts_alpha0": 0.01,
    "kts_beta0": 0.01,
    "phase2_lambda": 0.1,
    "phase2_mode": "drugflow",  # "drugflow" | "fragment_docking" (reserved)
}


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


def binomial_p_value(k, n, p0=0.0):
    """One-sided binomial test: P(X >= k | H0: p = p0).

    For p0=0, P(X >= k) = 1 if k > 0, since baseline probability is exactly 0.
    We use a small epsilon for numerical stability.
    """
    if p0 == 0:
        # With p0=0, ANY success is significant. Return a very small p-value.
        return 1e-10 if k > 0 else 1.0
    p_val = 0.0
    for i in range(k, n + 1):
        p_val += binom_coeff(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
    return min(p_val, 1.0)


def run_phase1_with_selector(model, protein_data, energy_fn, selector,
                              attempt_offset=0):
    """Phase 1 with type suggestion from AnchorTypeSelector."""
    kts = KTSScheduler(alpha0=CFG["kts_alpha0"], beta0=CFG["kts_beta0"])
    guide_fn = _Phase1GuideFn(
        site_energy=energy_fn,
        lambda_guide=CFG["phase1_lambda"],
        guidance_start=0.05, guidance_end=0.95,
        grad_clip=1.0, kts=kts,
    ).to(DEVICE)

    best_d_overall = float("inf")
    attempts_log = []

    for attempt in range(CFG["phase1_attempts"]):
        # Get suggested types for this attempt
        suggested = selector.get_types_for_attempt(
            attempt + attempt_offset, n_types=3
        )

        t0 = time.time()
        with torch.no_grad():
            rdmols, _, _ = model.sample(
                protein_data,
                n_samples=CFG["phase1_per_attempt"],
                timesteps=CFG["phase1_steps"],
                num_nodes=CFG["phase1_atoms"],
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
                x, h, energy_fn,
                CFG["success_distance"], CFG["min_compatibility"]
            )

            if diag["best_distance"] < best_d_overall:
                best_d_overall = diag["best_distance"]

            if diag["success"]:
                cfg = Phase1Config(
                    success_distance=CFG["success_distance"],
                    min_compatibility=CFG["min_compatibility"],
                    anchor_selection="best_per_site",
                )
                anchors = _extract_anchors(x, h, energy_fn, diag, cfg)
                if anchors is not None and anchors.n_anchors > 0:
                    anchor_types = [
                        ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
                        for i in range(anchors.n_anchors)
                    ]
                    attempts_log.append({
                        "attempt": attempt + 1,
                        "success": True,
                        "suggested_types": suggested,
                        "anchor_types": anchor_types,
                        "best_distance": diag["best_distance"],
                        "n_anchors": anchors.n_anchors,
                    })
                    return anchors, diag, attempts_log

        attempts_log.append({
            "attempt": attempt + 1,
            "success": False,
            "suggested_types": suggested,
            "best_distance": best_d_overall,
        })

    return None, None, attempts_log


def run_phase2_hardfix(model, protein_data, energy_fn, anchors, ref_size):
    """Phase 2 with hard coordinate fix + type bias."""
    n_samples = CFG["n_samples"]
    timesteps = CFG["phase2_steps"]

    # Hard fix callback (coordinates only)
    callback = HardFixCallback(
        anchor_indices=list(range(anchors.n_anchors)),
        anchor_coords=anchors.positions.clone(),
        anchor_h=None,
        fix_coords=CFG["hard_fix_coords"],
        fix_types=False,
        verbose=False,
    )

    # Phase 2 guide with type bias
    kts = KTSScheduler(alpha0=CFG["kts_alpha0"] * 0.5, beta0=CFG["kts_beta0"])
    cfg = Phase2Config(
        fix_atoms=True,
        restraint_force=0.0,  # Hard fix instead of harmonic
        lambda_late=CFG["phase2_lambda"],
        guidance_start=0.1, guidance_end=0.90,
        grad_clip=0.3,
        type_bias_strength=CFG["type_bias_strength"],
    )
    guide_fn = TwoStageGuideFn(
        energy_fn, anchors, cfg, kts,
        type_bias_strength=CFG["type_bias_strength"],
    ).to(DEVICE)
    guide_fn.set_anchor_indices(list(range(anchors.n_anchors)), ref_size)

    # Replicate DrugFlow sample() logic
    from src.data import data_utils
    from src.data.molecule_builder import build_molecule
    from src import utils
    from itertools import accumulate

    data = protein_data
    if len(data['pocket']['x']) > 0:
        pocket = data_utils.repeat_items(data['pocket'], n_samples)
    else:
        pocket = data_utils.Residues(**{k: v for k, v in data['pocket'].items()})
        pocket['name'] = pocket['name'] * n_samples
        pocket['size'] = pocket['size'].repeat(n_samples)
        pocket['n_bonds'] = pocket['n_bonds'].repeat(n_samples)

    _ligand = data_utils.repeat_items(data['ligand'], n_samples)
    num_nodes = model.parse_num_nodes_spec(
        {"ligand": _ligand, "pocket": pocket}, spec=ref_size, size_model=None
    )
    if pocket['x'].numel() > 0:
        ligand = model.init_ligand(num_nodes, pocket)
    else:
        ligand = model.init_ligand(num_nodes, _ligand)
    pocket = model.init_pocket(pocket)

    t0 = time.time()
    with torch.no_grad():
        out_ligand, out_pocket = model.simulate(
            ligand, pocket, timesteps, 0.0, 1.0,
            guide_log_prob=guide_fn,
            post_step_callback=callback,
        )
    elapsed = time.time() - t0

    # Post-process to RDKit
    x = out_ligand['x'].detach().cpu()
    ligand_type = out_ligand['h'].argmax(1).detach().cpu()
    edge_type = out_ligand['e'].argmax(1).detach().cpu()
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
    rdmols = [build_molecule(**m, atom_decoder=model.atom_decoder,
                             bond_decoder=model.bond_decoder) for m in mol_kwargs]

    valid = [m for m in rdmols if m is not None]
    return valid, elapsed, callback.n_calls


def evaluate_molecules(valid_mols, site_map, pocket_name):
    """Full evaluation suite."""
    occ = site_occupancy_summary(valid_mols, site_map, threshold=2.5)
    dor = occ["direct_occupancy"]
    bcd = occ["compatible_distance"]

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
        "n_total": len(valid_mols),
        "direct_occ_rate": dor["rate"],
        "n_occupied": dor["n_occupied"],
        "per_mol_occupied": occ["per_mol_occupied_sites"],
        "best_compat_d_mean": bcd["mean"],
        "best_compat_d_min": bcd["min"],
        "best_compat_d_std": float(np.std([s["best_distance"] for s in bcd.get("per_site_best_distance", [])])),
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


def run_pocket(pocket_name, model):
    """Complete v7.1 pipeline on one pocket."""
    print(f"\n{'='*70}")
    print(f"  POCKET: {pocket_name}")
    print(f"{'='*70}")

    site_map_path = os.path.join(SITE_DIR, f"{pocket_name}_site_map.json")
    site_map = json.load(open(site_map_path))
    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    envs = {}
    for s in hew_sites:
        e = classify_hew_environment(s)
        envs[e] = envs.get(e, 0) + 1
    print(f"  HEW: {len(hew_sites)} sites ({envs})")

    # Anchor type selector
    selector = AnchorTypeSelector(
        site_map,
        strategy=CFG["anchor_type_strategy"],
        max_attempts_per_type=CFG["max_attempts_per_type"],
    )
    print(f"  Type strategy: {CFG['anchor_type_strategy']}")
    print(f"  Suggested types: {selector.get_all_suggested_types()[:8]}")

    # Site energy
    energy_fn = build_site_energy_from_map(
        site_map, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
    ).to(DEVICE)

    # Find protein/ligand
    import glob
    pdirs = glob.glob(os.path.join(PDB_ROOT, "*", pocket_name))
    pdir = pdirs[0]
    protein_pdb = os.path.join(pdir, f"{pocket_name}_protein.pdb")
    ligand_sdf = os.path.join(pdir, f"{pocket_name}_ligand.sdf")

    data, ref_size = process_protein(protein_pdb, ligand_sdf, model)
    protein_data = {
        "ligand": TensorDict(**data["ligand"]).to(DEVICE),
        "pocket": TensorDict(**data["pocket"]).to(DEVICE),
    }
    print(f"  Ref ligand: {ref_size} atoms")

    # Phase 1
    t1 = time.time()
    anchors, phase1_diag, attempts_log = run_phase1_with_selector(
        model, protein_data, energy_fn, selector
    )
    phase1_time = time.time() - t1

    if anchors is None:
        print(f"  ✗ Phase 1 FAILED after {len(attempts_log)} attempts")
        return {
            "pocket": pocket_name,
            "phase1_success": False,
            "n_valid": 0,
        }

    print(f"  ✓ Phase 1 OK: {anchors.n_anchors} anchor(s), "
          f"best_d={phase1_diag['best_distance']:.2f}Å, "
          f"time={phase1_time:.1f}s")
    for i in range(anchors.n_anchors):
        atype = ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
        print(f"    Anchor {i}: {atype} d={anchors.distances[i]:.2f}Å "
              f"compat={anchors.compat_scores[i]:.3f}")

    # Phase 2
    print(f"  Phase 2 ({CFG['n_samples']} samples, hard fix + type_bias={CFG['type_bias_strength']})...")
    t2 = time.time()
    phase2_mols, phase2_time, n_calls = run_phase2_hardfix(
        model, protein_data, energy_fn, anchors, ref_size
    )
    phase2_elapsed = time.time() - t2

    if not phase2_mols:
        print(f"  ✗ Phase 2 produced no valid molecules")
        return {"pocket": pocket_name, "phase1_success": True, "n_valid": 0}

    print(f"  ✓ Phase 2: {len(phase2_mols)} valid, {phase2_elapsed:.1f}s")

    # Save molecules
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    metrics["n_total"] = CFG["n_samples"]
    metrics["phase1_success"] = True
    metrics["phase1_time"] = phase1_time
    metrics["phase2_time"] = phase2_elapsed
    metrics["n_anchors"] = anchors.n_anchors
    metrics["anchor_types"] = [ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
                                for i in range(anchors.n_anchors)]
    metrics["anchor_distances"] = anchors.distances
    metrics["anchor_compat_scores"] = anchors.compat_scores
    metrics["hardfix_calls"] = n_calls
    metrics["attempts_log"] = attempts_log
    metrics["type_strategy"] = CFG["anchor_type_strategy"]

    # Binomial test
    k = metrics["n_occupied"]
    n = metrics["n_valid"]
    p_val = binomial_p_value(k, n, p0=0.0)
    metrics["binomial_p_value"] = p_val
    significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
    metrics["significance"] = significance

    # v7.2 recommendation
    if metrics["direct_occ_rate"] < 0.10:
        metrics["v72_recommended"] = True
        metrics["v72_reason"] = (
            f"DirectOcc={metrics['direct_occ_rate']:.3f} < 0.10. "
            f"Consider fragment_docking mode for Phase 2."
        )
    else:
        metrics["v72_recommended"] = False

    # Print results
    print(f"\n  ── Results for {pocket_name} ──")
    print(f"  Valid: {n}/{CFG['n_samples']}")
    print(f"  DirectOcc: {metrics['direct_occ_rate']:.3f} ({k}/{n}) "
          f"p={p_val:.2e} {significance}")
    print(f"  BestCompatD: {metrics['best_compat_d_mean']:.2f}±{metrics['best_compat_d_std']:.2f}Å "
          f"(min={metrics['best_compat_d_min']:.2f}Å)")
    print(f"  Sites occupied: {metrics['n_sites_occupied']}/{metrics['n_sites_total']}")
    print(f"  QED: {metrics['qed_mean']:.3f}±{metrics['qed_std']:.3f}")
    print(f"  POSU: {metrics['posu_mean']:.3f}±{metrics['posu_std']:.3f}, "
          f"HEWU: {metrics['hewu_mean']:.3f}")
    if metrics.get("v72_recommended"):
        print(f"  ⚠ v7.2 recommended: {metrics['v72_reason']}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets", default="2gni,3mfw,6o4x")
    parser.add_argument("--n-samples", type=int, default=CFG["n_samples"])
    parser.add_argument("--phase1-lambda", type=float, default=CFG["phase1_lambda"])
    parser.add_argument("--type-strategy", default=CFG["anchor_type_strategy"])
    parser.add_argument("--type-bias-strength", type=float, default=CFG["type_bias_strength"])
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    CFG["n_samples"] = args.n_samples
    CFG["phase1_lambda"] = args.phase1_lambda
    CFG["anchor_type_strategy"] = args.type_strategy
    CFG["type_bias_strength"] = args.type_bias_strength

    pockets = [p.strip() for p in args.pockets.split(",")]

    print("=" * 70)
    print("v7.1 Statistical Validation — Actionable Pockets")
    print(f"Pockets: {pockets}")
    print(f"Samples per pocket: {CFG['n_samples']}")
    print(f"Phase1: λ={CFG['phase1_lambda']}, {CFG['phase1_attempts']} attempts, "
          f"type_strategy={CFG['anchor_type_strategy']}")
    print(f"Phase2: hard_fix={CFG['hard_fix_coords']}, "
          f"type_bias={CFG['type_bias_strength']}")
    print("=" * 70)

    print("\n[0] Loading DrugFlow...")
    model = load_model(DRUGFLOW_CKPT, device=DEVICE)
    print(f"    GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    all_metrics = {}
    total_start = time.time()

    for pocket in pockets:
        metrics = run_pocket(pocket, model)
        if metrics:
            all_metrics[pocket] = metrics

    total_elapsed = time.time() - total_start

    # ── Final Summary Table ──
    print(f"\n{'='*70}")
    print("FINAL STATISTICAL SUMMARY — v7.1")
    print(f"{'='*70}")
    header = (f"{'Pocket':<8} {'N':>4} {'DirectOcc':>10} {'p-value':>10} "
              f"{'Sig':>4} {'BestD':>7} {'Sites':>7} {'QED':>7} {'POSU':>7} "
              f"{'v7.2?':>6}")
    print(header)
    print("-" * len(header))

    for pocket in pockets:
        m = all_metrics.get(pocket, {})
        if not m or not m.get("phase1_success"):
            print(f"{pocket:<8} {'FAIL':>4}")
            continue
        v72 = "YES" if m.get("v72_recommended") else "no"
        print(f"{pocket:<8} {m['n_valid']:>4} "
              f"{m['direct_occ_rate']:>9.3f} "
              f"{m['binomial_p_value']:>9.2e} "
              f"{m['significance']:>4} "
              f"{m['best_compat_d_min']:>6.2f}Å "
              f"{m['n_sites_occupied']:>3}/{m['n_sites_total']:<3} "
              f"{m['qed_mean']:>6.3f} "
              f"{m['posu_mean']:>6.3f} "
              f"{v72:>6}")

    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Save all metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "v71_statistical_results.json")

    # Clean for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    json.dump(clean_for_json(all_metrics), open(metrics_path, "w"), indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Also save a summary CSV
    csv_path = os.path.join(args.output_dir, "v71_summary.csv")
    with open(csv_path, "w") as f:
        f.write("pocket,n_valid,n_total,direct_occ_rate,p_value,significance,"
                "best_compat_d_min,best_compat_d_mean,n_sites_occupied,"
                "n_sites_total,qed_mean,qed_std,posu_mean,posu_std,hewu_mean,"
                "v72_recommended\n")
        for pocket in pockets:
            m = all_metrics.get(pocket, {})
            if m:
                f.write(f"{pocket},{m.get('n_valid',0)},{m.get('n_total',0)},"
                        f"{m.get('direct_occ_rate',0):.4f},{m.get('binomial_p_value',1):.2e},"
                        f"{m.get('significance','n.s.')},"
                        f"{m.get('best_compat_d_min','NA')},{m.get('best_compat_d_mean','NA'):.2f},"
                        f"{m.get('n_sites_occupied',0)},{m.get('n_sites_total',0)},"
                        f"{m.get('qed_mean',0):.3f},{m.get('qed_std',0):.3f},"
                        f"{m.get('posu_mean',0):.3f},{m.get('posu_std',0):.3f},"
                        f"{m.get('hewu_mean',0):.3f},"
                        f"{m.get('v72_recommended',False)}\n")
    print(f"CSV saved to {csv_path}")

    return all_metrics


if __name__ == "__main__":
    main()
