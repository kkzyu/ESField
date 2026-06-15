#!/usr/bin/env python3
"""New evaluation metrics for KAG project (replaces legacy occupancy metrics).

Computes four categories of metrics for each generated molecule:

1. Centroid-to-HEW distances (min, avg per molecule)
2. Continuous Occupancy Score (COS) with sigma=1.5 Å
3. E_site energy from formula (1)
4. Per-condition statistics (mean, std, median) suitable for paper tables

Also generates:
  - Pareto frontier analysis (strain vs proximity)
  - Box plot / CDF data for plotting

Usage:
    python scripts/metrics_new.py \\
        --sdf-dir experiments/master_experiments/drugflow_main/3mfw/baseline/sdfs \\
        --site-map experiments/targetdiff_replication/site_maps/3mfw_site_map.json \\
        --strain-file experiments/master_experiments/drugflow_main/3mfw/baseline/meta.json \\
        --condition baseline \\
        --pocket 3mfw \\
        --output-dir results/metrics/3mfw_baseline

    # Or batch mode:
    python scripts/metrics_new.py --batch --results-root experiments/master_experiments/drugflow_main \\
        --site-maps-root experiments/targetdiff_replication/site_maps \\
        --output-dir results/metrics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ── Compatibility matrix (exact values from paper Appendix Table 10) ──
# 4 envs × 11 atom types (unknown, C_sp3, C_aromatic, N_donor, N_acceptor,
#                          O_acceptor, S, P, halogen, charged, B)
COMPAT_MATRIX = np.array([
    # Hydrophobic
    [-0.5,  1.0,  1.0, -0.8, -0.8, -0.8,  0.3, -0.5,  1.0, -1.0,  0.0],
    # Polar-unsatisfied
    [-0.5, -0.8, -0.5,  1.0,  1.0,  1.0, -0.3, -0.3, -0.8, -0.5,  0.0],
    # Mixed
    [-0.5,  0.5,  0.5,  0.3,  0.3,  0.3, -0.3, -0.3,  0.5, -0.5,  0.0],
    # Buried
    [-0.5,  0.8,  0.8, -0.3, -0.3, -0.3,  0.5, -0.3,  0.8, -0.8,  0.0],
], dtype=np.float64)

HEW_ENV_ORDER = ["hydrophobic", "polar_unsatisfied", "mixed", "buried"]

# RDKit atomic number → atom type index mapping
ATOMIC_NUM_TO_TYPE_IDX = {
    6: 1,   # C → C_sp3 (simplified; aromatic can be detected later)
    7: 3,   # N → N_donor
    8: 5,   # O → O_acceptor
    16: 6,  # S
    15: 7,  # P
    9: 8,   # F → halogen
    17: 8,  # Cl → halogen
    35: 8,  # Br → halogen
    53: 8,  # I → halogen
    5: 10,  # B
}


def classify_hew_environment(site: dict) -> str:
    """Classify a HEW site into one of four environment types."""
    features = site.get("features", {})
    hbond = features.get("hbond_count", 0)
    hydrophobic = features.get("hydrophobic_contact_count", 0)
    nearest_dist = features.get("nearest_protein_distance", 4.0)

    if nearest_dist < 2.5:
        return "buried"
    if hydrophobic >= 4 and hbond <= 1:
        return "hydrophobic"
    if hbond <= 1 and hydrophobic <= 2:
        return "polar_unsatisfied"
    return "mixed"


def env_to_idx(env: str) -> int:
    try:
        return HEW_ENV_ORDER.index(env)
    except ValueError:
        return 2  # default: mixed


def get_atom_type_idx(atom) -> int:
    """Map RDKit atom to our atom type index (one-hot, simplified)."""
    atomic_num = atom.GetAtomicNum()
    is_aromatic = atom.GetIsAromatic()
    if atomic_num == 6:
        return 2 if is_aromatic else 1  # C_aromatic or C_sp3
    return ATOMIC_NUM_TO_TYPE_IDX.get(atomic_num, 0)  # 0 = unknown


def compute_mol_centroid(mol) -> np.ndarray | None:
    """Compute the centroid (x,y,z) of a molecule."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    conf = mol.GetConformer()
    if conf is None:
        return None
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(mol.GetNumAtoms())])
    return coords.mean(axis=0)


def compute_centroid_hew_distances(
    centroid: np.ndarray,
    hew_centers: np.ndarray,
) -> dict[str, float]:
    """Compute min and average distance from centroid to HEW sites."""
    if len(hew_centers) == 0:
        return {"min_dist_centroid": float("inf"), "avg_dist_centroid": float("inf")}
    dists = np.linalg.norm(hew_centers - centroid[None, :], axis=1)
    return {
        "min_dist_centroid": float(dists.min()),
        "avg_dist_centroid": float(dists.mean()),
    }


def compute_cos(
    mol,
    hew_centers: np.ndarray,
    hew_env_indices: np.ndarray,
    sigma: float = 1.5,
) -> dict[str, float]:
    """Compute Continuous Occupancy Score (COS).

    For each HEW site k:
        COS_k = max_i [ exp(-d_ik²/(2σ²)) * Σ_a h_{i,a} * M_{e_k,a} ]

    where:
      - σ = 1.5 Å
      - d_ik = distance from atom i to site k
      - h_{i,a} = one-hot atom type (index a) for atom i
      - M = compatibility matrix
    """
    if mol is None or mol.GetNumAtoms() == 0 or len(hew_centers) == 0:
        return {"avg_COS": 0.0, "max_COS": 0.0, "per_site_COS": []}

    conf = mol.GetConformer()
    if conf is None:
        return {"avg_COS": 0.0, "max_COS": 0.0, "per_site_COS": []}

    n_atoms = mol.GetNumAtoms()
    n_sites = len(hew_centers)

    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z]
                       for i in range(n_atoms)])

    # Atom type one-hot [n_atoms, 11]
    h_onehot = np.zeros((n_atoms, 11), dtype=np.float64)
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx = get_atom_type_idx(atom)
        h_onehot[i, type_idx] = 1.0

    sigma2 = 2.0 * sigma ** 2
    per_site_cos = []

    for k in range(n_sites):
        # Distances from all atoms to site k
        rel = coords - hew_centers[k][None, :]  # [n_atoms, 3]
        dist_sq = (rel ** 2).sum(axis=1)  # [n_atoms]

        # Gaussian weight
        gauss = np.exp(-dist_sq / sigma2)  # [n_atoms]

        # Compatibility: h_onehot @ M[env_k, :].T = [n_atoms]
        env_idx = hew_env_indices[k]
        compat = h_onehot @ COMPAT_MATRIX[env_idx, :]  # [n_atoms]

        # COS_k = max_i [ exp(-d²/2σ²) * compat_i ]
        # Note: compat can be negative — use max of product
        scores = gauss * compat
        cos_k = float(scores.max()) if len(scores) > 0 else 0.0
        per_site_cos.append(cos_k)

    per_site_cos = np.array(per_site_cos)
    return {
        "avg_COS": float(per_site_cos.mean()) if len(per_site_cos) > 0 else 0.0,
        "max_COS": float(per_site_cos.max()) if len(per_site_cos) > 0 else 0.0,
        "per_site_COS": per_site_cos.tolist(),
    }


def compute_e_site(
    mol,
    hew_centers: np.ndarray,
    hew_env_indices: np.ndarray,
    sigma: float = 3.0,
    tau: float = 10.0,
) -> float:
    """Compute E_site from formula (1).

    E = -(1/τ) * log( Σ_i exp( τ * Σ_j compat_ij * exp(-d_ij²/(2σ²)) ) )

    Parameters:
      σ = 3.0 Å (default)
      τ = 10.0 (temperature)
    """
    if mol is None or mol.GetNumAtoms() == 0 or len(hew_centers) == 0:
        return 0.0

    conf = mol.GetConformer()
    if conf is None:
        return 0.0

    n_atoms = mol.GetNumAtoms()
    n_sites = len(hew_centers)

    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z]
                       for i in range(n_atoms)])

    # Atom type one-hot [n_atoms, 11]
    h_onehot = np.zeros((n_atoms, 11), dtype=np.float64)
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx = get_atom_type_idx(atom)
        h_onehot[i, type_idx] = 1.0

    sigma2 = 2.0 * sigma ** 2

    # Per-atom scores
    per_atom_scores = np.zeros(n_atoms, dtype=np.float64)
    for i in range(n_atoms):
        rel = hew_centers - coords[i][None, :]  # [n_sites, 3]
        dist_sq = (rel ** 2).sum(axis=1)  # [n_sites]
        gauss = np.exp(-dist_sq / sigma2)  # [n_sites]

        compat_i = h_onehot[i] @ COMPAT_MATRIX[hew_env_indices, :].T  # [n_sites]
        per_atom_scores[i] = (compat_i * gauss).sum()

    # Log-sum-exp with temperature τ
    tau_scores = tau * per_atom_scores
    # Numerical stability: subtract max
    max_score = tau_scores.max()
    e_site = -(1.0 / tau) * np.log(np.exp(tau_scores - max_score).sum()) - max_score / tau

    return float(e_site)


def compute_pareto_frontier(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Identify Pareto-optimal points (non-dominated set).

    For minimization in both x and y (lower = better).
    Point i dominates point j if x[i] <= x[j] AND y[i] <= y[j]
    AND at least one is strictly better.

    Returns:
        Boolean array, True for Pareto-optimal points.
    """
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                is_pareto[i] = False
                break
    return is_pareto


def evaluate_sdf_directory(
    sdf_dir: str | Path,
    site_map: dict,
    strain_values: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate all SDF molecules in a directory.

    Args:
        sdf_dir: path to directory containing .sdf files
        site_map: site map dict with HEW sites
        strain_values: optional list of strain values (one per molecule)

    Returns:
        dict with per-molecule metrics and aggregated statistics.
    """
    sdf_dir = Path(sdf_dir)
    sdf_files = sorted(sdf_dir.glob("*.sdf"))
    if not sdf_files:
        sdf_files = sorted(sdf_dir.glob("mol_*.sdf"))
    if not sdf_files:
        # Try reading a single combined SDF
        combined = sdf_dir / "_e.sdf"
        if combined.exists():
            sdf_files = [combined]

    # Extract HEW site info
    hew_sites = [s for s in site_map.get("sites", [])
                 if s.get("site_type") == "high_energy_water"]
    hew_centers = np.array([s["center"] for s in hew_sites])
    hew_env_indices = np.array([env_to_idx(classify_hew_environment(s))
                                 for s in hew_sites])

    all_mols = []
    for sf in sdf_files:
        if sf.stat().st_size == 0:
            continue  # skip empty files
        try:
            supplier = Chem.SDMolSupplier(str(sf), sanitize=False, removeHs=False)
        except OSError:
            continue  # skip corrupted files
        for mol in supplier:
            if mol is not None and mol.GetNumAtoms() > 0:
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                                     ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception:
                    pass
                all_mols.append(mol)

    if not all_mols:
        return {"error": f"No valid molecules found in {sdf_dir}", "n_mols": 0}

    # Per-molecule metrics
    results = []
    for mol_idx, mol in enumerate(all_mols):
        centroid = compute_mol_centroid(mol)
        centroid_dists = compute_centroid_hew_distances(centroid, hew_centers) if centroid is not None else {}
        cos = compute_cos(mol, hew_centers, hew_env_indices, sigma=1.5)
        e_site = compute_e_site(mol, hew_centers, hew_env_indices, sigma=3.0, tau=10.0)

        entry = {
            "mol_id": mol_idx,
            "n_atoms": mol.GetNumAtoms(),
            **centroid_dists,
            **cos,
            "E_site": e_site,
        }
        if strain_values is not None and mol_idx < len(strain_values):
            entry["strain"] = strain_values[mol_idx]
        results.append(entry)

    # ── Aggregate statistics ──
    metrics_keys = ["min_dist_centroid", "avg_dist_centroid", "avg_COS",
                    "max_COS", "E_site"]
    if strain_values is not None:
        metrics_keys.append("strain")

    stats = {}
    for key in metrics_keys:
        vals = [r[key] for r in results if key in r and r[key] is not None
                and not (isinstance(r[key], float) and math.isinf(r[key]))]
        if vals:
            stats[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }

    # ── Pareto frontier analysis ──
    pareto_info = {}
    if strain_values is not None:
        strains = np.array([r.get("strain", float("nan")) for r in results])
        cos_vals = np.array([r.get("avg_COS", float("nan")) for r in results])
        min_dists = np.array([r.get("min_dist_centroid", float("nan")) for r in results])

        # Filter NaN
        valid = ~(np.isnan(strains) | np.isnan(cos_vals) | np.isnan(min_dists))
        if valid.sum() >= 2:
            # Pareto: strain vs -min_dist_centroid (lower strain, lower distance = better)
            x_strain = strains[valid]
            y_prox = min_dists[valid]  # lower = closer to HEW
            pareto_mask = compute_pareto_frontier(x_strain, y_prox)
            pareto_info["strain_vs_proximity"] = {
                "n_total": int(valid.sum()),
                "n_pareto": int(pareto_mask.sum()),
                "pareto_fraction": float(pareto_mask.sum() / valid.sum()),
                "pareto_indices": np.where(valid)[0][pareto_mask].tolist(),
            }

            # Also COS-strain pareto
            y_cos = -cos_vals[valid]  # higher COS = better, negate for minimization
            pareto_mask_cos = compute_pareto_frontier(x_strain, y_cos)
            pareto_info["strain_vs_COS"] = {
                "n_total": int(valid.sum()),
                "n_pareto": int(pareto_mask_cos.sum()),
                "pareto_fraction": float(pareto_mask_cos.sum() / valid.sum()),
            }

    return {
        "n_mols": len(all_mols),
        "n_hew_sites": len(hew_centers),
        "per_mol": results,
        "statistics": stats,
        "pareto": pareto_info,
    }


def write_outputs(results: dict, output_dir: Path, condition: str, pocket: str):
    """Write JSON summary and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON summary ──
    json_path = output_dir / f"{pocket}_{condition}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ JSON summary → {json_path}")

    # ── Per-molecule CSVs ──
    if results.get("per_mol"):
        per_mol = results["per_mol"]

        # centroid distances
        csv_centroid = output_dir / f"{pocket}_{condition}_centroid_dists.csv"
        with open(csv_centroid, "w") as f:
            f.write("mol_id,min_dist_centroid,avg_dist_centroid\n")
            for r in per_mol:
                f.write(f"{r['mol_id']},{r.get('min_dist_centroid','')},{r.get('avg_dist_centroid','')}\n")

        # COS
        csv_cos = output_dir / f"{pocket}_{condition}_COS.csv"
        with open(csv_cos, "w") as f:
            f.write("mol_id,avg_COS,max_COS\n")
            for r in per_mol:
                f.write(f"{r['mol_id']},{r.get('avg_COS','')},{r.get('max_COS','')}\n")

        # E_site
        csv_esite = output_dir / f"{pocket}_{condition}_Esite.csv"
        with open(csv_esite, "w") as f:
            f.write("mol_id,E_site\n")
            for r in per_mol:
                f.write(f"{r['mol_id']},{r.get('E_site','')}\n")

        print(f"  ✓ CSV files → {output_dir}")

    # ── Statistics summary ──
    if results.get("statistics"):
        stats_path = output_dir / f"{pocket}_{condition}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(results["statistics"], f, indent=2)
        print(f"  ✓ Statistics → {stats_path}")


def generate_pareto_plot(
    all_results: dict[str, dict],  # key = "pocket/condition"
    output_path: Path,
    y_metric: str = "avg_COS",
):
    """Generate a Pareto scatter plot comparing conditions.

    Args:
        all_results: dict mapping "pocket/condition" → evaluate_sdf_directory output
        output_path: path for the output PNG
        y_metric: "avg_COS" or "min_dist_centroid"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping Pareto plot.")
        return

    colors = {"baseline": "#1f77b4", "hard_fix": "#ff7f0e",
              "com_projection": "#2ca02c", "unguided": "#1f77b4",
              "KAG": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, res in all_results.items():
        if res.get("per_mol") is None:
            continue
        per_mol = res["per_mol"]
        strains = [r.get("strain", float("nan")) for r in per_mol]
        if y_metric == "avg_COS":
            y_vals = [r.get("avg_COS", float("nan")) for r in per_mol]
        else:
            y_vals = [r.get("min_dist_centroid", float("nan")) for r in per_mol]

        valid = [(s, y) for s, y in zip(strains, y_vals)
                 if not (np.isnan(s) or np.isnan(y))]
        if not valid:
            continue
        sx, sy = zip(*valid)
        color = colors.get(label.split("/")[-1], "#333333")
        ax.scatter(sx, sy, alpha=0.6, label=label, color=color, s=20)

    ax.set_xlabel("Strain (kcal/mol)")
    ax.set_ylabel(y_metric)
    ax.set_title(f"Pareto: Strain vs {y_metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Pareto plot → {output_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="KAG New Metrics Evaluation")
    parser.add_argument("--sdf-dir", help="Directory with SDF files")
    parser.add_argument("--site-map", help="Site map JSON file")
    parser.add_argument("--strain-file", help="JSON with strain values per mol")
    parser.add_argument("--condition", default="unknown", help="Condition label")
    parser.add_argument("--pocket", default="unknown", help="Pocket ID")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: scan results-root for all conditions")
    parser.add_argument("--results-root", help="Root of experiment results")
    parser.add_argument("--site-maps-root", help="Root of site maps")
    args = parser.parse_args()

    if args.batch:
        if not args.results_root or not args.site_maps_root:
            parser.error("--batch requires --results-root and --site-maps-root")
        run_batch(args)
    else:
        if not args.sdf_dir or not args.site_map:
            parser.error("--sdf-dir and --site-map required (or use --batch)")
        run_single(args)


def run_single(args):
    if not HAS_RDKIT:
        print("ERROR: rdkit required. pip install rdkit")
        sys.exit(1)

    site_map = json.loads(Path(args.site_map).read_text())

    strain_values = None
    if args.strain_file:
        strain_data = json.loads(Path(args.strain_file).read_text())
        if isinstance(strain_data, list):
            strain_values = strain_data
        elif isinstance(strain_data, dict) and "per_mol_strains" in strain_data:
            strain_values = strain_data["per_mol_strains"]

    results = evaluate_sdf_directory(args.sdf_dir, site_map, strain_values)
    results["condition"] = args.condition
    results["pocket"] = args.pocket

    output_dir = Path(args.output_dir)
    write_outputs(results, output_dir, args.condition, args.pocket)

    # Print summary
    stats = results.get("statistics", {})
    print(f"\n  Summary for {args.pocket}/{args.condition} ({results['n_mols']} molecules, "
          f"{results['n_hew_sites']} HEW sites):")
    for key, s in stats.items():
        print(f"    {key:<25}  mean={s['mean']:>10.4f}  std={s['std']:>10.4f}  median={s['median']:>10.4f}")

    if results.get("pareto"):
        for pk, pv in results["pareto"].items():
            print(f"    Pareto {pk}: {pv['n_pareto']}/{pv['n_total']} = {pv['pareto_fraction']:.1%}")


def run_batch(args):
    """Batch-process all pockets and conditions."""
    results_root = Path(args.results_root)
    site_maps_root = Path(args.site_maps_root)

    all_results = {}

    for pocket_dir in sorted(results_root.iterdir()):
        if not pocket_dir.is_dir():
            continue
        pocket = pocket_dir.name

        # Load site map
        site_map_path = site_maps_root / f"{pocket}_site_map.json"
        if not site_map_path.exists():
            print(f"  ⚠ No site map for {pocket}, skipping.")
            continue
        site_map = json.loads(site_map_path.read_text())

        for cond_dir in sorted(pocket_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            condition = cond_dir.name
            sdf_dir = cond_dir / "sdfs"
            if not sdf_dir.exists():
                # Check for combined SDF
                sdf_files = list(cond_dir.glob("*.sdf"))
                if sdf_files:
                    sdf_dir = cond_dir
                else:
                    print(f"  ⚠ No SDFs for {pocket}/{condition}, skipping.")
                    continue

            # Load strain values if available
            strain_values = None
            meta_path = cond_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if "per_mol_strains" in meta:
                        strain_values = meta["per_mol_strains"]
                except Exception:
                    pass

            print(f"\nProcessing {pocket}/{condition} ...")
            results = evaluate_sdf_directory(sdf_dir, site_map, strain_values)
            results["condition"] = condition
            results["pocket"] = pocket

            output_dir = Path(args.output_dir) / pocket / condition
            write_outputs(results, output_dir, condition, pocket)
            all_results[f"{pocket}/{condition}"] = results

            # Print summary
            stats = results.get("statistics", {})
            print(f"  {results['n_mols']} mols, {results['n_hew_sites']} HEW sites")
            for key, s in stats.items():
                print(f"    {key:<25} {s['mean']:>10.4f} ± {s['std']:>8.4f}")

    # Generate combined Pareto plots per pocket
    for pocket in sorted(set(k.split("/")[0] for k in all_results)):
        pocket_results = {k: v for k, v in all_results.items()
                          if k.startswith(pocket + "/")}
        if len(pocket_results) >= 2:
            output_path = Path(args.output_dir) / pocket / "pareto_strain_vs_proximity.png"
            generate_pareto_plot(pocket_results, output_path, y_metric="min_dist_centroid")

            output_path_cos = Path(args.output_dir) / pocket / "pareto_strain_vs_COS.png"
            generate_pareto_plot(pocket_results, output_path_cos, y_metric="avg_COS")

    # All-pockets combined
    if len(all_results) >= 2:
        output_path = Path(args.output_dir) / "pareto_strain_vs_proximity_all.png"
        generate_pareto_plot(all_results, output_path, y_metric="min_dist_centroid")


if __name__ == "__main__":
    main()
