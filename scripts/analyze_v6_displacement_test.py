#!/usr/bin/env python3
"""Analyze ESField mechanism test results (v5 or v6-D).

Computes per-condition metrics across multiple pockets:
  - POSU, HEWU, SWScore, HCFU (v2.1)
  - HEW occupancy score, mean compatible distance to HEW center
  - QED, SA, MW, logP, HBD, HBA, rotatable bonds
  - Validity rate, clash rate, integrity factor
  - Wrong-pocket gap, random-matrix ablation gap

Usage:
  # Analyze v5 mechanism test
  python scripts/analyze_v6_displacement_test.py \
    --experiment-dir experiments/pdbbind_water_sites/v5_mechanism_test \
    --label "V5 Mechanism Test"

  # Analyze v6-D displacement test
  python scripts/analyze_v6_displacement_test.py \
    --experiment-dir experiments/pdbbind_water_sites/v6_displacement_test \
    --label "V6-D Displacement Test"

  # With custom pockets
  python scripts/analyze_v6_displacement_test.py \
    --experiment-dir experiments/pdbbind_water_sites/v5_sum_norm_test \
    --pockets 1sle,2clh,3mfw
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors

from evaluation.posu import (
    compute_posu, classify_hew_environment, is_compatible_hew_v2,
)
from utils.chemistry import infer_atom_type, atomic_number, normalize_element
from utils.geometry import distance as calc_distance

DEFAULT_SITE_MAPS = ROOT / "experiments/pdbbind_water_sites/test_sites"
DEFAULT_POCKETS_JSON = ROOT / "experiments/pdbbind_water_sites/test_pockets.json"


def load_sdf_molecules(sdf_path):
    """Load molecules from SDF, return valid rdkit mols."""
    if not Path(sdf_path).exists():
        return []
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    mols = []
    for m in suppl:
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            mols.append(m)
        except Exception:
            continue
    return mols


def compute_molecule_metrics(mol, site_map):
    """Compute all quality and site-utilization metrics for one molecule."""
    metrics = {"valid": 1}

    try:
        metrics["qed"] = QED.qed(mol)
    except Exception:
        metrics["qed"] = 0.0

    try:
        from rdkit.Contrib.SA_Score import sascorer
        metrics["sa"] = sascorer.calculateScore(mol)
    except Exception:
        metrics["sa"] = 5.0

    try:
        metrics["mw"] = Descriptors.MolWt(mol)
        metrics["logp"] = Descriptors.MolLogP(mol)
    except Exception:
        metrics["mw"] = 0.0
        metrics["logp"] = 0.0

    try:
        metrics["hbd"] = rdMolDescriptors.CalcNumHBD(mol)
        metrics["hba"] = rdMolDescriptors.CalcNumHBA(mol)
        metrics["rot_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    except Exception:
        metrics["hbd"] = 0
        metrics["hba"] = 0
        metrics["rot_bonds"] = 0

    try:
        posu_result = compute_posu(mol, site_map)
        metrics["posu"] = posu_result["posu"]
        metrics["hewu"] = posu_result["hew_mean"]
        metrics["sw_total"] = posu_result.get("sw_total", 1.0)
        metrics["hc_mean"] = posu_result.get("hc_mean", 0.0)
        metrics["integrity"] = posu_result.get("integrity_factor", 1.0)
    except Exception:
        metrics["posu"] = metrics["hewu"] = metrics["hc_mean"] = 0.0
        metrics["sw_total"] = 1.0
        metrics["integrity"] = 0.0

    try:
        metrics.update(_compute_hew_occupancy(mol, site_map))
    except Exception:
        metrics.update({"hew_occ": 0.0, "hew_dist_compat": 10.0,
                       "hew_n_occupied": 0, "hew_n_total": 0})

    try:
        metrics["clash_rate"] = _compute_intra_clash(mol)
    except Exception:
        metrics["clash_rate"] = 0.0

    return metrics


def _compute_hew_occupancy(mol, site_map):
    """HEW-specific: max occupancy and compatible-atom distances."""
    if isinstance(site_map, (str, Path)):
        site_map = json.loads(Path(site_map).read_text())

    hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
    if not hew_sites:
        return {"hew_occ": 0.0, "hew_dist_compat": 10.0,
                "hew_n_occupied": 0, "hew_n_total": 0}

    conf = mol.GetConformer()
    atoms = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(a.GetIdx())
        atoms.append({
            "coord": (pos.x, pos.y, pos.z),
            "atom_type": infer_atom_type(a.GetSymbol()),
            "atomic_number": atomic_number(a.GetSymbol()),
            "element": normalize_element(a.GetSymbol()),
        })

    sigma_occ = 1.2
    max_occ = 0.0
    compat_dists = []
    n_occupied = 0

    for site in hew_sites:
        site_center = tuple(site["center"])
        sigma = max(0.5, site.get("radius", 1.4)) * sigma_occ
        best_occ = 0.0
        best_compat_dist = None

        for atom in atoms:
            d = calc_distance(atom["coord"], site_center)
            occ = np.exp(-d**2 / (2 * sigma**2))
            if occ > best_occ:
                best_occ = occ
                if is_compatible_hew_v2(atom["atom_type"], atom["atomic_number"], site):
                    best_compat_dist = d

        max_occ = max(max_occ, best_occ)
        if best_occ > 0.1:
            n_occupied += 1
        if best_compat_dist is not None:
            compat_dists.append(best_compat_dist)

    return {
        "hew_occ": max_occ,
        "hew_dist_compat": float(np.mean(compat_dists)) if compat_dists else 10.0,
        "hew_n_occupied": n_occupied,
        "hew_n_total": len(hew_sites),
    }


def _compute_intra_clash(mol):
    """Rate of intra-ligand atom pairs closer than 1.2 Angstrom."""
    conf = mol.GetConformer()
    n_clash = 0
    n_pairs = 0
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            d = ((pi.x - pj.x)**2 + (pi.y - pj.y)**2 + (pi.z - pj.z)**2)**0.5
            n_pairs += 1
            if d < 1.2:
                n_clash += 1
    return n_clash / max(n_pairs, 1)


def _aggregate_metrics(all_metrics):
    """Aggregate per-molecule metrics into per-condition statistics."""
    agg = {"n_mols": len(all_metrics)}
    for key in ["qed", "sa", "mw", "logp", "posu", "hewu", "sw_total",
                "hc_mean", "integrity", "hew_occ", "hew_dist_compat",
                "clash_rate", "hbd", "hba", "rot_bonds"]:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    agg["valid_rate"] = len(all_metrics) / max(agg["n_mols"], 1)
    agg["hew_n_occupied_mean"] = float(np.mean(
        [m.get("hew_n_occupied", 0) for m in all_metrics]))
    return agg


def auto_detect_conditions(experiment_dir, pockets):
    """Discover conditions by scanning subdirectory names."""
    conds = defaultdict(set)
    exp = Path(experiment_dir)
    if not exp.exists():
        return {}
    for pid in pockets:
        for subdir in exp.iterdir():
            if subdir.is_dir() and subdir.name.startswith(pid):
                cond_name = subdir.name[len(pid) + 1:]  # strip "{pid}_"
                conds[pid].add(cond_name)
    return {pid: sorted(cs) for pid, cs in conds.items()}


def analyze_experiment(experiment_dir, pockets, site_maps_dir,
                       wrong_pocket_map=None, label="Experiment"):
    """Run full analysis across all pockets and conditions."""
    exp_dir = Path(experiment_dir)
    site_dir = Path(site_maps_dir)

    # Auto-detect conditions
    all_conds = auto_detect_conditions(exp_dir, pockets)
    if not all_conds:
        print(f"No subdirectories found in {exp_dir}")
        return {}

    results = {}
    for pid in pockets:
        pid_conds = all_conds.get(pid, [])
        correct_site = str(site_dir / "correct" / f"{pid}_site_map.json")

        for cond in pid_conds:
            key = f"{pid}_{cond}"
            sdf_path = exp_dir / key / "molecules.sdf"
            print(f"  [{pid}] {cond} ... ", end="", flush=True)

            mols = load_sdf_molecules(sdf_path)
            if not mols:
                print("NO MOLS")
                results[key] = {"n_mols": 0, "error": "No valid molecules"}
                continue

            site_map = json.loads(Path(correct_site).read_text())
            per_mol = [compute_molecule_metrics(m, site_map) for m in mols]
            results[key] = _aggregate_metrics(per_mol)
            print(f"{len(mols)} mols, POSU={results[key].get('posu_mean', 0):.3f}, "
                  f"QED={results[key].get('qed_mean', 0):.3f}")

    return results


def print_summary(results, pockets, wrong_pocket_map=None, label="Experiment"):
    """Print per-condition summary with pass/fail diagnostics."""
    cond_groups = defaultdict(list)
    for key, metrics in results.items():
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        pid, cond = parts[0], parts[1]
        cond_groups[cond].append((pid, metrics))

    print(f"\n{'='*80}")
    print(f"{label.upper()} — SUMMARY")
    print(f"{'='*80}")

    # --- Per-condition table ---
    for cond_name, items in sorted(cond_groups.items()):
        metrics_list = [m for _, m in items if m.get("n_mols", 0) > 0]
        if not metrics_list:
            print(f"\n{cond_name}: NO DATA")
            continue

        print(f"\n--- {cond_name} ({len(metrics_list)} pockets, "
              f"{sum(m['n_mols'] for m in metrics_list)} mols) ---")

        rows = [
            ("POSU", "posu_mean"),
            ("HEWU", "hewu_mean"),
            ("HEW Occ", "hew_occ_mean"),
            ("d_HEW compat", "hew_dist_compat_mean"),
            ("QED", "qed_mean"),
            ("SA", "sa_mean"),
            ("logP", "logp_mean"),
            ("Clash rate", "clash_rate_mean"),
            ("Valid rate", "valid_rate"),
        ]
        for label, key in rows:
            vals = [m.get(key, 0) for m in metrics_list if key in m]
            if vals:
                print(f"  {label:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # --- Ablation gaps ---
    if wrong_pocket_map:
        print("\n--- Ablation Gaps ---")
        for pid in pockets:
            correct_key = _find_key(results, pid, "lambda1.0")
            wrong_pid = wrong_pocket_map.get(pid, "")
            wrong_key = _find_key(results, pid, f"wrong_{wrong_pid}")

            c_posu = results.get(correct_key, {}).get("posu_mean", 0) if correct_key else 0
            w_posu = results.get(wrong_key, {}).get("posu_mean", 0) if wrong_key else 0
            wrong_gap = c_posu - w_posu if c_posu else 0
            print(f"  {pid}: correct={c_posu:.3f}, wrong={w_posu:.3f}, gap={wrong_gap:+.3f}")

    # --- Pass/fail diagnostics ---
    print("\n--- Diagnostics ---")

    base_keys = {pid: _find_key(results, pid, "baseline") for pid in pockets}
    # Find the main test condition (v5_lambda1.0, v6d_lambda1.0, etc.)
    test_cond_candidates = ["v5_lambda1.0", "v6d_lambda1.0", "v5-sum_lambda1.0",
                           "v5-sumnorm_lambda1.0"]
    test_keys = {}
    for pid in pockets:
        for cand in test_cond_candidates:
            k = f"{pid}_{cand}"
            if k in results and results[k].get("n_mols", 0) > 0:
                test_keys[pid] = k
                break

    if test_keys:
        # 1. Improvement over baseline
        n_improved = 0
        for pid in pockets:
            bk = base_keys.get(pid)
            tk = test_keys.get(pid)
            if bk and tk:
                base_posu = results[bk].get("posu_mean", 0)
                test_posu = results[tk].get("posu_mean", 0)
                if test_posu > base_posu:
                    n_improved += 1
                print(f"  {pid}: baseline={base_posu:.3f}, test={test_posu:.3f} "
                      f"({'+' if test_posu > base_posu else '-'}{abs(test_posu-base_posu):.3f})")
        print(f"  Improved over baseline: {n_improved}/{len(pockets)}")

        # 2. QED check
        base_qed = np.mean([results.get(base_keys[pid], {}).get("qed_mean", 0)
                           for pid in pockets if base_keys.get(pid)])
        test_qed = np.mean([results.get(test_keys[pid], {}).get("qed_mean", 0)
                           for pid in pockets if test_keys.get(pid)])
        print(f"  QED: baseline={base_qed:.3f}, test={test_qed:.3f} "
              f"({'OK' if test_qed > 0.3 else 'COLLAPSED'})")

        # 3. HEW distance
        base_dhew = np.mean([results.get(base_keys[pid], {}).get("hew_dist_compat_mean", 10)
                            for pid in pockets if base_keys.get(pid)])
        test_dhew = np.mean([results.get(test_keys[pid], {}).get("hew_dist_compat_mean", 10)
                            for pid in pockets if test_keys.get(pid)])
        print(f"  d_HEW_compat: baseline={base_dhew:.2f}A, test={test_dhew:.2f}A "
              f"({'closer' if test_dhew < base_dhew else 'no improvement'})")


def _find_key(results, pid, pattern):
    """Find a result key matching {pid}_{pattern}."""
    for key in results:
        if key.startswith(f"{pid}_") and pattern in key:
            return key
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ESField mechanism test results (v5 or v6-D)")
    parser.add_argument("--experiment-dir", required=True,
                        help="Directory containing per-condition subdirectories")
    parser.add_argument("--pockets", default="3ohi,2clh,3mfw,4bis,1sle",
                        help="Comma-separated pocket IDs")
    parser.add_argument("--site-maps-dir", default=str(DEFAULT_SITE_MAPS),
                        help="Directory containing correct/random/shuffled site maps")
    parser.add_argument("--wrong-pocket-map", default=None,
                        help="JSON mapping pocket->wrong_pocket for ablation")
    parser.add_argument("--label", default="Experiment",
                        help="Label for report header")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    pockets = [p.strip() for p in args.pockets.split(",")]

    wrong_map = None
    if args.wrong_pocket_map:
        wrong_map = json.loads(Path(args.wrong_pocket_map).read_text())
    else:
        # Default wrong-pocket map
        wrong_map = {
            "3ohi": "5g60", "2clh": "1sle", "3mfw": "2jkr",
            "4bis": "3r01", "1sle": "2wgi",
        }

    results = analyze_experiment(
        args.experiment_dir, pockets, args.site_maps_dir,
        wrong_pocket_map=wrong_map, label=args.label)

    if results:
        print_summary(results, pockets, wrong_pocket_map=wrong_map,
                     label=args.label)

    # Save results
    out_path = args.output or (Path(args.experiment_dir) / "analysis.json")
    serializable = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
        for k, v in results.items()
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
