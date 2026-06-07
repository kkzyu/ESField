#!/usr/bin/env python3
"""v7.1 Ablation Study — 10 pockets × 6 conditions.

Tests the contribution of each v7.1 component:
  1. v7.1_full          — all components
  2. random_anchor_types — random vs suggested types
  3. no_type_bias        — type_bias_strength = 0
  4. soft_restraint      — harmonic restraint vs hard fix
  5. lambda_phase1_2.5   — weaker Phase 1 guidance
  6. lambda_phase1_10.0  — stronger Phase 1 guidance

Optimization: conditions sharing the same Phase 1 settings reuse anchors
to avoid redundant expensive Phase 1 runs.

Run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v71_ablation.py
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
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

ESFIELD_ROOT = "/root/ESField"
for p in [f"{ESFIELD_ROOT}/src", ESFIELD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from guidance.latent_guidance import (
    build_site_energy_from_map, classify_hew_environment, ATOM_TYPE_VOCAB,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    Phase1Config, _Phase1GuideFn, _compute_diagnostics, _extract_anchors,
    _tensors_from_rdmol, AnchorAtoms, TwoStageGuideFn, Phase2Config,
    AnchorTypeSelector,
)
from guidance.hard_fix import patch_drugflow_hardfix, HardFixCallback
from evaluation.site_occupancy import site_occupancy_summary
from evaluation.posu import compute_posu

patch_drugflow_hardfix()

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
PDB_ROOT = "/root/autodl-tmp/data/PDB/P-L"
SITE_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/site_maps"
OUTPUT_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_ablation"
DEVICE = "cuda:0"

# ── Condition definitions ──
# Grouped by Phase 1 settings to enable anchor reuse
CONDITIONS = [
    {"name": "v7.1_full",           "type_strategy": "suggested", "type_bias": 0.3,
     "hard_fix": True,  "restraint": 0.0,  "p1_lambda": 5.0, "p1_group": "A"},
    {"name": "random_anchor_types",  "type_strategy": "random",    "type_bias": 0.3,
     "hard_fix": True,  "restraint": 0.0,  "p1_lambda": 5.0, "p1_group": "A"},
    {"name": "no_type_bias",         "type_strategy": "suggested", "type_bias": 0.0,
     "hard_fix": True,  "restraint": 0.0,  "p1_lambda": 5.0, "p1_group": "A"},
    {"name": "soft_restraint",       "type_strategy": "suggested", "type_bias": 0.3,
     "hard_fix": False, "restraint": 10.0, "p1_lambda": 5.0, "p1_group": "A"},
    {"name": "lambda_phase1_2.5",    "type_strategy": "suggested", "type_bias": 0.3,
     "hard_fix": True,  "restraint": 0.0,  "p1_lambda": 2.5, "p1_group": "B"},
    {"name": "lambda_phase1_10.0",   "type_strategy": "suggested", "type_bias": 0.3,
     "hard_fix": True,  "restraint": 0.0,  "p1_lambda": 10.0, "p1_group": "C"},
]

N_SAMPLES = 25
PHASE1_ATTEMPTS = 2
PHASE1_PER_ATTEMPT = 5
PHASE1_ATOMS = 4
PHASE1_STEPS = 100
PHASE2_STEPS = 100

KTS_ALPHA0 = 0.01
KTS_BETA0 = 0.01


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


def run_phase1(model, protein_data, energy_fn, selector, p1_lambda):
    """Phase 1 with given type selector and lambda."""
    kts = KTSScheduler(alpha0=KTS_ALPHA0, beta0=KTS_BETA0)
    guide_fn = _Phase1GuideFn(
        site_energy=energy_fn, lambda_guide=p1_lambda,
        guidance_start=0.05, guidance_end=0.95,
        grad_clip=1.0, kts=kts,
    ).to(DEVICE)

    for attempt in range(PHASE1_ATTEMPTS):
        with torch.no_grad():
            rdmols, _, _ = model.sample(
                protein_data, n_samples=PHASE1_PER_ATTEMPT,
                timesteps=PHASE1_STEPS, num_nodes=PHASE1_ATOMS,
                guide_log_prob=guide_fn,
            )
        for mol in rdmols:
            if mol is None: continue
            x, h = _tensors_from_rdmol(mol, device=DEVICE)
            if x is None: continue
            diag = _compute_diagnostics(x, h, energy_fn, 2.5, -0.5)
            if diag["success"]:
                anchors = _extract_anchors(x, h, energy_fn, diag,
                    Phase1Config(success_distance=2.5, min_compatibility=-0.5,
                                 anchor_selection="best_per_site"))
                if anchors and anchors.n_anchors > 0:
                    return anchors, diag
    return None, None


def run_phase2(model, protein_data, energy_fn, anchors, ref_size, cond):
    """Phase 2 with condition-specific settings."""
    n = N_SAMPLES
    ts = PHASE2_STEPS

    # Build guide
    kts = KTSScheduler(alpha0=KTS_ALPHA0 * 0.5, beta0=KTS_BETA0)
    cfg = Phase2Config(
        fix_atoms=cond["hard_fix"], restraint_force=cond["restraint"],
        lambda_late=0.1, guidance_start=0.1, guidance_end=0.90,
        grad_clip=0.3, type_bias_strength=cond["type_bias"],
    )
    guide_fn = TwoStageGuideFn(energy_fn, anchors, cfg, kts,
                               type_bias_strength=cond["type_bias"]).to(DEVICE)
    guide_fn.set_anchor_indices(list(range(anchors.n_anchors)), ref_size)

    # Hard fix callback (only if hard_fix enabled)
    callback = None
    if cond["hard_fix"]:
        callback = HardFixCallback(
            anchor_indices=list(range(anchors.n_anchors)),
            anchor_coords=anchors.positions.clone(),
            anchor_h=None, fix_coords=True, fix_types=False, verbose=False,
        )

    # Setup
    from src.data import data_utils
    from src.data.molecule_builder import build_molecule
    from src import utils
    from itertools import accumulate

    data = protein_data
    if len(data['pocket']['x']) > 0:
        pocket = data_utils.repeat_items(data['pocket'], n)
    else:
        pocket = data_utils.Residues(**{k: v for k, v in data['pocket'].items()})
        pocket['name'] = pocket['name'] * n; pocket['size'] = pocket['size'].repeat(n)
        pocket['n_bonds'] = pocket['n_bonds'].repeat(n)

    _ligand = data_utils.repeat_items(data['ligand'], n)
    num_nodes = model.parse_num_nodes_spec(
        {"ligand": _ligand, "pocket": pocket}, spec=ref_size, size_model=None)
    if pocket['x'].numel() > 0:
        ligand = model.init_ligand(num_nodes, pocket)
    else:
        ligand = model.init_ligand(num_nodes, _ligand)
    pocket = model.init_pocket(pocket)

    with torch.no_grad():
        out_lig, out_poc = model.simulate(
            ligand, pocket, ts, 0.0, 1.0,
            guide_log_prob=guide_fn, post_step_callback=callback,
        )

    # Post-process
    x = out_lig['x'].detach().cpu(); lt = out_lig['h'].argmax(1).detach().cpu()
    et = out_lig['e'].argmax(1).detach().cpu(); lm = ligand['mask'].detach().cpu()
    lb = ligand['bonds'].detach().cpu(); lem = ligand['edge_mask'].detach().cpu()
    sizes = torch.unique(ligand['mask'], return_counts=True)[1].tolist()
    offsets = list(accumulate(sizes[:-1], initial=0))
    mk = {'coords': utils.batch_to_list(x, lm),
          'atom_types': utils.batch_to_list(lt, lm),
          'bonds': utils.batch_to_list_for_indices(lb, lem, offsets),
          'bond_types': utils.batch_to_list(et, lem)}
    mk = [{k: v[i] for k, v in mk.items()} for i in range(len(mk['coords']))]
    rdmols = [build_molecule(**m, atom_decoder=model.atom_decoder,
                             bond_decoder=model.bond_decoder) for m in mk]
    return [m for m in rdmols if m is not None]


def evaluate(valid_mols, site_map):
    """Compute metrics for a set of molecules."""
    occ = site_occupancy_summary(valid_mols, site_map, threshold=2.5)
    qeds, posu_vals, hewu_vals = [], [], []
    for m in valid_mols:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            qeds.append(QED.qed(m))
            p = compute_posu(m, site_map)
            posu_vals.append(p["posu"]); hewu_vals.append(p["hew_mean"])
        except: pass
    return {
        "n_valid": len(valid_mols),
        "direct_occ_rate": occ["direct_occupancy"]["rate"],
        "n_occupied": occ["direct_occupancy"]["n_occupied"],
        "best_compat_d_min": occ["compatible_distance"]["min"],
        "best_compat_d_mean": occ["compatible_distance"]["mean"],
        "n_sites_occupied": occ["compatible_distance"]["n_sites_occupied"],
        "n_sites_total": occ["compatible_distance"]["n_sites_total"],
        "qed_mean": float(np.mean(qeds)) if qeds else 0,
        "qed_std": float(np.std(qeds)) if qeds else 0,
        "posu_mean": float(np.mean(posu_vals)) if posu_vals else 0,
        "posu_std": float(np.std(posu_vals)) if posu_vals else 0,
        "hewu_mean": float(np.mean(hewu_vals)) if hewu_vals else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets-file",
                        default=f"{OUTPUT_DIR}/pocket_ids_10.txt")
    parser.add_argument("--conditions", default=None,
                        help="Comma-separated condition names (default: all 6)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    # Load pockets
    with open(args.pockets_file) as f:
        pockets = [line.strip() for line in f if line.strip()]
    print(f"Pockets: {len(pockets)} — {pockets}")

    # Filter conditions
    conds = CONDITIONS
    if args.conditions:
        wanted = set(args.conditions.split(","))
        conds = [c for c in CONDITIONS if c["name"] in wanted]
    print(f"Conditions: {len(conds)} — {[c['name'] for c in conds]}")

    print(f"\n[0] Loading DrugFlow...")
    model = load_model(DRUGFLOW_CKPT, device=DEVICE)

    all_results = {}
    total_start = time.time()

    for pocket_idx, pocket_name in enumerate(pockets):
        print(f"\n{'='*60}")
        print(f"  [{pocket_idx+1}/{len(pockets)}] {pocket_name}")
        print(f"{'='*60}")

        # Load site map
        sm = json.load(open(os.path.join(SITE_DIR, f"{pocket_name}_site_map.json")))
        energy_fn = build_site_energy_from_map(sm, sigma_distance=3.0,
            enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed")).to(DEVICE)

        # Protein
        import glob
        pdir = glob.glob(os.path.join(PDB_ROOT, "*", pocket_name))[0]
        data, ref_size = process_protein(
            os.path.join(pdir, f"{pocket_name}_protein.pdb"),
            os.path.join(pdir, f"{pocket_name}_ligand.sdf"), model)
        protein_data = {"ligand": TensorDict(**data["ligand"]).to(DEVICE),
                        "pocket": TensorDict(**data["pocket"]).to(DEVICE)}

        # Cache Phase 1 results by group
        p1_cache = {}  # group -> (anchors, diag)
        pocket_results = {}

        for ci, cond in enumerate(conds):
            p1_group = cond["p1_group"]
            cond_name = cond["name"]

            # Phase 1: use cache if available
            if p1_group in p1_cache:
                anchors, p1_diag = p1_cache[p1_group]
                if anchors is None:
                    pocket_results[cond_name] = {"phase1_success": False, "n_valid": 0}
                    continue
            else:
                # Build type selector
                selector = AnchorTypeSelector(sm, strategy=cond["type_strategy"],
                                              max_attempts_per_type=2)
                t0 = time.time()
                anchors, p1_diag = run_phase1(model, protein_data, energy_fn,
                                              selector, cond["p1_lambda"])
                p1_elapsed = time.time() - t0
                p1_cache[p1_group] = (anchors, p1_diag)

                if anchors is None:
                    print(f"  {cond_name}: Phase1 FAILED ({p1_elapsed:.0f}s)")
                    pocket_results[cond_name] = {"phase1_success": False, "n_valid": 0}
                    continue
                atypes = [ATOM_TYPE_VOCAB[anchors.type_indices[i].item()]
                          for i in range(anchors.n_anchors)]
                print(f"  [{p1_group}] Phase1 OK ({p1_elapsed:.0f}s): "
                      f"{anchors.n_anchors} anchors [{','.join(atypes)}] "
                      f"d={p1_diag['best_distance']:.2f}Å")

            # Phase 2
            t0 = time.time()
            mols = run_phase2(model, protein_data, energy_fn, anchors, ref_size, cond)
            p2_elapsed = time.time() - t0

            if not mols:
                pocket_results[cond_name] = {"phase1_success": True, "n_valid": 0}
                continue

            metrics = evaluate(mols, sm)
            metrics["phase1_success"] = True
            metrics["n_total"] = N_SAMPLES
            metrics["p2_elapsed"] = p2_elapsed
            metrics["n_anchors"] = anchors.n_anchors
            metrics["cond_config"] = cond
            pocket_results[cond_name] = metrics

            print(f"    {cond_name}: {metrics['n_valid']} valid, "
                  f"DirOcc={metrics['direct_occ_rate']:.2f}, "
                  f"QED={metrics['qed_mean']:.2f}, "
                  f"POSU={metrics['posu_mean']:.3f}, "
                  f"({p2_elapsed:.0f}s)")

        all_results[pocket_name] = pocket_results

        # Save incremental results
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Final Summary ──
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}")

    # Per-condition mean across all pockets
    cond_summary = {}
    for cond in conds:
        cn = cond["name"]
        rates, qeds, posus = [], [], []
        for p in pockets:
            pr = all_results.get(p, {}).get(cn, {})
            if pr.get("n_valid", 0) > 0:
                rates.append(pr["direct_occ_rate"])
                qeds.append(pr["qed_mean"])
                posus.append(pr["posu_mean"])
        cond_summary[cn] = {
            "n_pockets_ok": len(rates),
            "mean_occ_rate": np.mean(rates) if rates else 0,
            "std_occ_rate": np.std(rates) if rates else 0,
            "mean_qed": np.mean(qeds) if qeds else 0,
            "mean_posu": np.mean(posus) if posus else 0,
        }

    # Print table
    header = f"{'Condition':<25} {'N OK':>5} {'OccRate':>8} {'QED':>6} {'POSU':>6}"
    print(header)
    print("-" * len(header))
    for cond in conds:
        cs = cond_summary[cond["name"]]
        print(f"{cond['name']:<25} {cs['n_pockets_ok']:>5} "
              f"{cs['mean_occ_rate']:>7.3f}±{cs['std_occ_rate']:.3f} "
              f"{cs['mean_qed']:>5.3f} {cs['mean_posu']:>5.3f}")

    # Per-pocket per-condition detail
    print(f"\n{'='*80}")
    print("PER-POCKET DETAIL")
    print(f"{'='*80}")
    col_w = 10
    header2 = f"{'Pocket':<8} " + " ".join(f"{c['name']:<{col_w}}" for c in conds)
    print(header2)
    print("-" * len(header2))
    for p in pockets:
        row = f"{p:<8} "
        for cond in conds:
            pr = all_results.get(p, {}).get(cond["name"], {})
            if pr.get("n_valid", 0) > 0:
                row += f"{pr['direct_occ_rate']:.3f}".ljust(col_w)
            else:
                row += "FAIL".ljust(col_w)
        print(row)

    # Save results
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        return obj

    out = {"conditions": [{k: v for k, v in c.items()} for c in conds],
           "pockets": pockets, "n_samples": N_SAMPLES,
           "cond_summary": cond_summary,
           "results": all_results,
           "total_elapsed": total_elapsed}
    json.dump(clean(out), open(f"{args.output_dir}/ablation_results.json", "w"), indent=2)
    print(f"\nResults saved to {args.output_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
