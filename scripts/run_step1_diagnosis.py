#!/usr/bin/env python3
"""DEPRECATED — use analyze_experiments.py --step 1 instead.

Kept for reference. The new analyze_experiments.py works with the
experiment_matrix output structure from run_experiment_matrix.py.
"""

import json, sys, re, numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.posu import compute_posu
from evaluation.site_blindness import compute_sbr, compute_sqd


def main():
    results_dir = ROOT / "experiments/pdbbind_water_sites/guided"
    test_sites_dir = ROOT / "experiments/pdbbind_water_sites/test_sites/correct"

    # Find all baseline SDFs
    baseline_dirs = sorted(results_dir.glob("*_baseline"))
    if not baseline_dirs:
        # Only have 6dvm_baseline
        baseline_dirs = [results_dir / "6dvm_baseline"]

    all_sbr = []
    all_sqd = []
    all_posu = []

    print(f"{'Pocket':<12} {'POSU':>8} {'SBR':>8} {'SQD':>8} {'r(Vina,POSU)':>14} {'GoodMols':>10}")
    print("-" * 70)

    for bdir in baseline_dirs:
        pocket_id = bdir.name.replace("_baseline", "")
        sdf_path = bdir / "molecules.sdf"
        if not sdf_path.exists():
            continue

        site_map_path = test_sites_dir / f"{pocket_id}_site_map.json"
        if not site_map_path.exists():
            # Try training site maps
            site_map_path = ROOT / f"experiments/pdbbind_water_sites/site_maps/{pocket_id}_site_map.json"
        if not site_map_path.exists():
            print(f"{pocket_id:<12} SKIP (no site map)")
            continue

        site_map = json.load(open(site_map_path))
        mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False))

        # Collect Vina scores if available
        vina_scores = []
        docking_dir = ROOT / f"experiments/pdbbind_water_sites/guided/docking/{pocket_id}_baseline"
        if docking_dir.exists():
            for out_file in sorted(docking_dir.glob("out_*.pdbqt")):
                with open(out_file) as f:
                    for line in f:
                        m = re.search(r'REMARK VINA RESULT:\s+(-?\d+\.\d+)', line)
                        if m:
                            vina_scores.append(float(m.group(1)))
                            break

        # Pad
        while len(vina_scores) < len(mols):
            vina_scores.append(None)

        # SBR
        sbr_result = compute_sbr(mols, site_map, vina_scores=vina_scores, qed_threshold=0.4)
        sqd_result = compute_sqd(mols, site_map, vina_scores)

        # Mean POSU
        posu_vals = []
        for m in mols:
            if m is None: continue
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                posu_vals.append(compute_posu(m, site_map)["posu"])
            except Exception: pass

        mean_posu = np.mean(posu_vals) if posu_vals else float("nan")

        print(f"{pocket_id:<12} {mean_posu:>8.4f} {sbr_result['sbr']:>8.1%} {sqd_result['sqd']:>8.4f} {sqd_result['spearman_r']:>14.4f} {sbr_result['n_good']:>10}")

        all_sbr.append(sbr_result['sbr'])
        all_sqd.append(sqd_result['sqd'])
        all_posu.append(mean_posu)

    print(f"\nSummary ({len(all_sbr)} pockets):")
    print(f"  Mean SBR: {np.mean(all_sbr):.1%} ± {np.std(all_sbr):.1%}")
    print(f"  Mean SQD: {np.mean(all_sqd):.4f} ± {np.std(all_sqd):.4f}")
    print(f"  Mean POSU: {np.mean(all_posu):.4f} ± {np.std(all_posu):.4f}")


if __name__ == "__main__":
    main()
