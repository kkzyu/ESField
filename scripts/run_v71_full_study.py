#!/usr/bin/env python3
"""v7.1 Full Study — Baseline + Ablation + Diversity + Statistics.

Generates:
  1. Unconditional DrugFlow baseline (30 molecules × 10 pockets)
  2. v7.1_full generation (50 molecules × 10 pockets)
  3. Diversity metrics (Vendi, Tanimoto) for all conditions
  4. Binomial significance tests vs baseline (DirectOcc = 0)
  5. Clopper-Pearson 95% confidence intervals
  6. Comprehensive CSV output

Supports --pocket for single-pocket runs and --all for batch.

Run from DrugFlow directory:
    cd /root/baselines/DrugFlow/code/DrugFlow-main && \
    python /root/ESField/scripts/run_v71_full_study.py --all
"""

import json, os, sys, time, warnings, argparse, csv
from pathlib import Path
from math import comb as binom_coeff

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
from evaluation.diversity import compute_diversity_metrics

patch_drugflow_hardfix()

DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
PDB_ROOT = "/root/autodl-tmp/data/PDB/P-L"
SITE_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/site_maps"
OUTPUT_DIR = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_full_study"
POCKETS_FILE = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_ablation/pocket_ids_10.txt"
DEVICE = "cuda:0"

N_BASELINE = 30
N_V7 = 50
PHASE1_LAMBDA = 5.0
PHASE1_STEPS = 100
PHASE1_ATOMS = 4
PHASE2_STEPS = 100


def load_model(ckpt, dev="cuda:0"):
    warnings.filterwarnings("ignore")
    _o = torch.load
    torch.load = lambda *a, **kw: _o(*a, **{**kw, "weights_only": False})
    try:
        m = lmod.DrugFlow.load_from_checkpoint(ckpt, map_location=dev)
    finally:
        torch.load = _o
    return m.to(dev).eval()


def process_protein(ppdb, rlig, model):
    from Bio.PDB import PDBParser
    pm = PDBParser(QUIET=True).get_structure("", ppdb)[0]
    rm = Chem.SDMolSupplier(rlig)[0]; rs = rm.GetNumAtoms()
    lr, pr = process_raw_pair(pm, rm, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation, compute_nerf_params=True)
    lr["name"] = "ligand"
    c = partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None)
    d = next(iter(DataLoader([{"ligand": lr, "pocket": pr}], batch_size=1, collate_fn=c)))
    return d, rs


def clopper_pearson_ci(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial confidence interval."""
    from scipy.stats import beta as beta_dist
    if n == 0: return (0.0, 1.0)
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def binomial_p_value(k, n, p0=0.0):
    """One-sided binomial: P(X >= k | H0: p = p0)."""
    if p0 == 0: return 1e-10 if k > 0 else 1.0
    pv = sum(binom_coeff(n, i) * (p0**i) * ((1-p0)**(n-i)) for i in range(k, n+1))
    return min(pv, 1.0)


def generate_baseline(model, prot_data, ref_size, n_samples):
    """Generate unconditional DrugFlow molecules."""
    t0 = time.time()
    with torch.no_grad():
        rdmols, _, _ = model.sample(prot_data, n_samples=n_samples,
                                     timesteps=100, num_nodes=ref_size)
    return [m for m in rdmols if m is not None], time.time() - t0


def generate_v7(model, prot_data, energy_fn, sm, ref_size, n_samples):
    """Full v7.1 pipeline (Phase1 + Phase2 hard fix)."""
    selector = AnchorTypeSelector(sm, strategy="suggested", max_attempts_per_type=2)
    # Phase 1
    kts = KTSScheduler(alpha0=0.01, beta0=0.01)
    p1_guide = _Phase1GuideFn(energy_fn, PHASE1_LAMBDA, 0.05, 0.95, 1.0, kts).to(DEVICE)
    anchors = None
    for _ in range(2):
        with torch.no_grad():
            rdmols, _, _ = model.sample(prot_data, n_samples=5, timesteps=PHASE1_STEPS,
                                         num_nodes=PHASE1_ATOMS, guide_log_prob=p1_guide)
        for mol in rdmols:
            if mol is None: continue
            x, h = _tensors_from_rdmol(mol, device=DEVICE)
            if x is None: continue
            diag = _compute_diagnostics(x, h, energy_fn, 2.5, -0.5)
            if diag["success"]:
                anchors = _extract_anchors(x, h, energy_fn, diag,
                    Phase1Config(success_distance=2.5, min_compatibility=-0.5,
                                 anchor_selection="best_per_site"))
                if anchors: break
        if anchors: break
    if anchors is None: return [], 0, None

    # Phase 2
    cb = HardFixCallback(list(range(anchors.n_anchors)), anchors.positions.clone(),
                         None, fix_coords=True, fix_types=False)
    kts2 = KTSScheduler(alpha0=0.005, beta0=0.01)
    cfg = Phase2Config(fix_atoms=True, restraint_force=0.0, lambda_late=0.1,
                       guidance_start=0.1, guidance_end=0.90, grad_clip=0.3,
                       type_bias_strength=0.3)
    p2_guide = TwoStageGuideFn(energy_fn, anchors, cfg, kts2, type_bias_strength=0.3).to(DEVICE)
    p2_guide.set_anchor_indices(list(range(anchors.n_anchors)), ref_size)

    from src.data import data_utils
    from src.data.molecule_builder import build_molecule
    from src import utils
    from itertools import accumulate

    data = prot_data; n = n_samples
    if len(data['pocket']['x']) > 0:
        pocket = data_utils.repeat_items(data['pocket'], n)
    else:
        pocket = data_utils.Residues(**{k: v for k, v in data['pocket'].items()})
        pocket.update(name=pocket['name']*n, size=pocket['size'].repeat(n),
                      n_bonds=pocket['n_bonds'].repeat(n))
    _lig = data_utils.repeat_items(data['ligand'], n)
    nn = model.parse_num_nodes_spec({"ligand": _lig, "pocket": pocket}, spec=ref_size)
    ligand = model.init_ligand(nn, pocket) if pocket['x'].numel() > 0 else model.init_ligand(nn, _lig)
    pocket = model.init_pocket(pocket)

    t0 = time.time()
    with torch.no_grad():
        ol, op = model.simulate(ligand, pocket, PHASE2_STEPS, 0.0, 1.0,
                                guide_log_prob=p2_guide, post_step_callback=cb)
    elapsed = time.time() - t0

    xo = ol['x'].detach().cpu(); lt = ol['h'].argmax(1).detach().cpu()
    et = ol['e'].argmax(1).detach().cpu(); lm = ligand['mask'].detach().cpu()
    lb = ligand['bonds'].detach().cpu(); le = ligand['edge_mask'].detach().cpu()
    sz = torch.unique(ligand['mask'], return_counts=True)[1].tolist()
    offs = list(accumulate(sz[:-1], initial=0))
    mk = {'coords': utils.batch_to_list(xo, lm), 'atom_types': utils.batch_to_list(lt, lm),
          'bonds': utils.batch_to_list_for_indices(lb, le, offs),
          'bond_types': utils.batch_to_list(et, le)}
    mk = [{k: v[i] for k, v in mk.items()} for i in range(len(mk['coords']))]
    rdmols = [build_molecule(**m, atom_decoder=model.atom_decoder,
                             bond_decoder=model.bond_decoder) for m in mk]
    return [m for m in rdmols if m is not None], elapsed, anchors


def compute_metrics(valid_mols, site_map):
    """Full metrics suite."""
    occ = site_occupancy_summary(valid_mols, site_map, threshold=2.5)
    div = compute_diversity_metrics(valid_mols)
    qeds, posus, hewus = [], [], []
    for m in valid_mols:
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            qeds.append(QED.qed(m))
            p = compute_posu(m, site_map); posus.append(p["posu"]); hewus.append(p["hew_mean"])
        except: pass
    k = occ["direct_occupancy"]["n_occupied"]; n = len(valid_mols)
    ci_lo, ci_hi = clopper_pearson_ci(k, n) if n > 0 else (0.0, 1.0)
    p_val = binomial_p_value(k, n, 0.0) if n > 0 else 1.0
    return {
        "n_valid": n, "n_occupied": k, "direct_occ_rate": occ["direct_occupancy"]["rate"],
        "ci_95_lo": ci_lo, "ci_95_hi": ci_hi, "p_value_vs_zero": p_val,
        "best_compat_d_min": occ["compatible_distance"]["min"],
        "best_compat_d_mean": occ["compatible_distance"]["mean"],
        "n_sites_occupied": occ["compatible_distance"]["n_sites_occupied"],
        "n_sites_total": occ["compatible_distance"]["n_sites_total"],
        "qed_mean": float(np.mean(qeds)) if qeds else 0,
        "qed_std": float(np.std(qeds)) if qeds else 0,
        "posu_mean": float(np.mean(posus)) if posus else 0,
        "posu_std": float(np.std(posus)) if posus else 0,
        "hewu_mean": float(np.mean(hewus)) if hewus else 0,
        "vendi_score": div["vendi_score"],
        "mean_pairwise_tanimoto": div["mean_pairwise_tanimoto"],
    }


def run_pocket(pocket_name, model):
    """Generate baseline + v7.1 for one pocket."""
    print(f"\n{'='*60}\n  {pocket_name}\n{'='*60}")
    import glob
    sm = json.load(open(os.path.join(SITE_DIR, f"{pocket_name}_site_map.json")))
    energy_fn = build_site_energy_from_map(sm, sigma_distance=3.0,
        enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed")).to(DEVICE)
    pdir = glob.glob(os.path.join(PDB_ROOT, "*", pocket_name))[0]
    data, ref_size = process_protein(
        os.path.join(pdir, f"{pocket_name}_protein.pdb"),
        os.path.join(pdir, f"{pocket_name}_ligand.sdf"), model)
    pd = {"ligand": TensorDict(**data["ligand"]).to(DEVICE),
          "pocket": TensorDict(**data["pocket"]).to(DEVICE)}

    results = {}

    # Baseline
    print(f"  Baseline ({N_BASELINE} samples)...")
    b_mols, b_time = generate_baseline(model, pd, ref_size, N_BASELINE)
    b_metrics = compute_metrics(b_mols, sm)
    b_metrics["generation_time"] = b_time
    results["baseline"] = b_metrics
    print(f"    {b_metrics['n_valid']} valid, DirOcc={b_metrics['direct_occ_rate']:.3f}, "
          f"Vendi={b_metrics['vendi_score']:.1f}, QED={b_metrics['qed_mean']:.2f}, "
          f"({b_time:.0f}s)")

    # v7.1
    print(f"  v7.1 ({N_V7} samples)...")
    v7_mols, v7_time, anchors = generate_v7(model, pd, energy_fn, sm, ref_size, N_V7)
    if v7_mols:
        v7_metrics = compute_metrics(v7_mols, sm)
        v7_metrics["generation_time"] = v7_time
        v7_metrics["phase1_success"] = True
        v7_metrics["n_anchors"] = anchors.n_anchors if anchors else 0
        results["v7.1"] = v7_metrics
        print(f"    {v7_metrics['n_valid']} valid, DirOcc={v7_metrics['direct_occ_rate']:.3f} "
              f"({v7_metrics['ci_95_lo']:.3f}-{v7_metrics['ci_95_hi']:.3f}), "
              f"Vendi={v7_metrics['vendi_score']:.1f}, QED={v7_metrics['qed_mean']:.2f}, "
              f"p={v7_metrics['p_value_vs_zero']:.2e}, ({v7_time:.0f}s)")
    else:
        results["v7.1"] = {"n_valid": 0, "phase1_success": False}

    # Save molecules
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for label, mols in [("baseline", b_mols), ("v7.1", v7_mols)]:
        if not mols: continue
        sdf = os.path.join(OUTPUT_DIR, f"{pocket_name}_{label}.sdf")
        w = Chem.SDWriter(sdf); w.SetKekulize(False)
        for m in mols:
            try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except: pass
            w.write(m)
        w.close()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", default=None, help="Single pocket name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pockets-file", default=POCKETS_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.pocket:
        pockets = [args.pocket]
    elif args.all:
        pockets = [l.strip() for l in open(args.pockets_file) if l.strip()]
    else:
        parser.print_help(); return

    print(f"v7.1 Full Study: {len(pockets)} pockets")
    print(f"Baseline: {N_BASELINE} samples, v7.1: {N_V7} samples each\n")

    model = load_model(DRUGFLOW_CKPT)
    all_results = {}

    for i, p in enumerate(pockets):
        print(f"\n[{i+1}/{len(pockets)}]", end="")
        all_results[p] = run_pocket(p, model)

    # ── Save combined CSV ──
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "full_study_results.csv")
    fields = ["pocket", "condition", "n_valid", "n_occupied", "direct_occ_rate",
              "ci_95_lo", "ci_95_hi", "p_value_vs_zero", "best_compat_d_min",
              "best_compat_d_mean", "n_sites_occupied", "n_sites_total",
              "qed_mean", "qed_std", "posu_mean", "posu_std", "hewu_mean",
              "vendi_score", "mean_pairwise_tanimoto", "generation_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for pkt, conds in all_results.items():
            for cond, metrics in conds.items():
                row = {"pocket": pkt, "condition": cond}
                row.update({k: v for k, v in metrics.items() if k in fields})
                w.writerow(row)

    print(f"\nCSV saved to {csv_path}")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print(f"{'Pocket':<8} {'Cond':<10} {'DirOcc':>8} {'95% CI':>16} {'p-val':>8} "
          f"{'Vendi':>6} {'QED':>6} {'POSU':>6}")
    print("-" * 70)
    for pkt, conds in all_results.items():
        for cond in ["baseline", "v7.1"]:
            m = conds.get(cond, {})
            if not m: continue
            ci = f"[{m.get('ci_95_lo',0):.3f}, {m.get('ci_95_hi',0):.3f}]"
            print(f"{pkt:<8} {cond:<10} {m.get('direct_occ_rate',0):>7.3f} "
                  f"{ci:>16} {m.get('p_value_vs_zero',1):>8.2e} "
                  f"{m.get('vendi_score',0):>5.1f} {m.get('qed_mean',0):>5.2f} "
                  f"{m.get('posu_mean',0):>5.3f}")

    # Save JSON
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list): return [clean(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    json.dump(clean(all_results), open(os.path.join(args.output_dir, "full_study_results.json"), "w"), indent=2)
    print(f"\nJSON saved.")


if __name__ == "__main__":
    main()
