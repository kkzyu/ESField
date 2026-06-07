#!/usr/bin/env python3
"""Comprehensive ESField guided vs baseline analysis + visualizations.

Metrics computed:
  1. Nearest site distance (per-atom → nearest site, mean/std/max)
  2. Site energy (compatibility potential score, lower = better)
  3. Site occupancy (fraction of atoms within cutoff of each site)
  4. Atom type distribution (histogram comparison)
  5. Per-site spatial analysis

Visualizations:
  1. 3D scatter: sites + ligand atoms (guided vs baseline)
  2. Energy distribution histogram
  3. Site occupancy heatmap
  4. Per-site distance comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAFLOW_ROOT = Path("/root/PAFlow-main")

# Import ESField modules FIRST (while ESField path dominates)
sys.path.insert(0, str(ROOT / "src"))
from models.potential_network import CompatibilityPotential, PotentialConfig  # noqa: E402

# Then add PAFlow paths on top for PAFlow-specific imports
sys.path.insert(0, str(PAFLOW_ROOT / "scripts"))
sys.path.insert(0, str(PAFLOW_ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ============================================================
# Metric computations
# ============================================================

def nearest_site_distance(pos, site_centers):
    """Mean distance from each atom to its nearest site center (Angstrom).

    Calculation: For each ligand atom position, compute Euclidean distance
    to all site centers, take the minimum. Report mean, std, max across atoms.

    Lower = atoms are closer to sites. This is a pure geometry metric.
    """
    pos_t = torch.tensor(pos, dtype=torch.float32)
    centers_t = torch.tensor(site_centers, dtype=torch.float32)
    if len(pos_t) == 0 or len(centers_t) == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "median": float("nan")}
    dists = torch.cdist(pos_t, centers_t)
    min_dists = dists.min(dim=1).values.numpy()
    return {
        "mean": float(np.mean(min_dists)),
        "std": float(np.std(min_dists)),
        "max": float(np.max(min_dists)),
        "median": float(np.median(min_dists)),
    }


def site_energy(pos, atom_types, site_map, potential, max_atom_type=11):
    """Mean compatibility energy per atom (lower = more compatible).

    Calculation: For each (atom, site) pair, compute E(atom_type, site_type,
    distance, relative_position, site_radius, site_confidence) through the
    trained potential network. Weight by Gaussian(distance, sigma_site) *
    site_confidence. Sum weighted energies across sites for each atom.

    Energy is negative for compatible pairs (e.g., hydrophobic atom near
    hydrophobic cavity), positive for incompatible pairs. Lower total
    energy means better overall compatibility.
    """
    centers = torch.tensor([s["center"] for s in site_map["sites"]], dtype=torch.float32)
    radii = torch.tensor([s["radius"] for s in site_map["sites"]], dtype=torch.float32)
    confs = torch.tensor([s["confidence"] for s in site_map["sites"]], dtype=torch.float32)
    stype_map = {"unknown": 0, "high_energy_water": 1, "stable_water": 2, "hydrophobic_cavity": 3}
    stypes = torch.tensor([stype_map.get(s["site_type"], 0) for s in site_map["sites"]], dtype=torch.long)

    n_atoms = len(pos)
    n_sites = len(centers)
    if n_sites == 0 or n_atoms == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "per_atom": []}

    pos_t = torch.tensor(pos, dtype=torch.float32) if not isinstance(pos, torch.Tensor) else pos.float()
    at_t = torch.tensor(atom_types, dtype=torch.long).clamp(0, max_atom_type - 1) if not isinstance(atom_types, torch.Tensor) else atom_types.long().clamp(0, max_atom_type - 1)

    rel = pos_t[:, None, :] - centers[None, :, :]
    dist = torch.norm(rel, dim=-1).clamp_min(1e-8)
    sigma = radii.clamp_min(1e-4)
    weight = torch.exp(-dist**2 / (2 * sigma[None, :]**2)) * confs[None, :]

    per_atom_energies = []
    for at_idx in at_t.unique().tolist():
        mask = at_t == at_idx
        n_masked = mask.sum().item()
        if n_masked == 0:
            continue
        at_flat = torch.full((n_masked * n_sites,), at_idx, dtype=torch.long)
        e_flat = potential(
            at_flat,
            stypes.repeat(n_masked),
            rel[mask].reshape(-1, 3),
            dist[mask].reshape(-1),
            radii.repeat(n_masked),
            confs.repeat(n_masked),
        )
        e_grid = e_flat.reshape(n_masked, n_sites)
        atom_energy = (e_grid * weight[mask]).sum(dim=1)
        per_atom_energies.append(atom_energy.detach().numpy())

    if per_atom_energies:
        all_energies = np.concatenate(per_atom_energies)
        return {
            "mean": float(np.mean(all_energies)),
            "std": float(np.std(all_energies)),
            "min": float(np.min(all_energies)),
            "max": float(np.max(all_energies)),
            "per_atom": all_energies.tolist(),
        }
    return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "per_atom": []}


def site_occupancy(pos, site_map, cutoff_mult=1.5):
    """Fraction of atoms within cutoff*radius of each site.

    Calculation: For each site s with radius r_s, count atoms whose distance
    to s < cutoff_mult * r_s. Report occupancy per site.

    Higher occupancy = more atoms placed near that site.
    """
    centers = np.array([s["center"] for s in site_map["sites"]])
    radii = np.array([s["radius"] for s in site_map["sites"]])
    pos_a = np.array(pos) if not isinstance(pos, np.ndarray) else pos

    occupancies = {}
    for i, s in enumerate(site_map["sites"]):
        dists = np.linalg.norm(pos_a - centers[i], axis=1)
        cutoff = radii[i] * cutoff_mult
        occ = np.mean(dists < cutoff)
        occupancies[s.get("site_id", f"site_{i}")] = {
            "occupancy": float(occ),
            "site_type": s["site_type"],
            "radius": float(radii[i]),
        }
    return occupancies


def atom_type_distribution(atom_types, max_types=11):
    """Histogram of atom types in generated molecules.

    Reports count and fraction per atom type index.
    """
    at = np.array(atom_types).flatten()
    at_clipped = np.clip(at, 0, max_types - 1)
    counts = np.bincount(at_clipped, minlength=max_types)
    total = counts.sum()
    return {f"type_{i}": {"count": int(c), "fraction": float(c / total if total > 0 else 0)} for i, (c, count_val) in enumerate(zip(counts, counts)) if count_val > 0}


def per_site_distance(pos, site_map):
    """Mean distance from all atoms to each individual site."""
    centers = np.array([s["center"] for s in site_map["sites"]])
    pos_a = np.array(pos) if not isinstance(pos, np.ndarray) else pos
    result = {}
    for i, s in enumerate(site_map["sites"]):
        dists = np.linalg.norm(pos_a - centers[i], axis=1)
        result[s.get("site_id", f"site_{i}")] = {
            "mean": float(np.mean(dists)),
            "median": float(np.median(dists)),
            "min": float(np.min(dists)),
            "site_type": s["site_type"],
        }
    return result


# ============================================================
# Visualizations
# ============================================================

def plot_3d_scatter(pos_b, pos_g, site_map, output_path, protein_id):
    """3D scatter: baseline (blue) vs guided (red) ligand atoms + sites (green spheres)."""
    fig = plt.figure(figsize=(14, 6))

    # Baseline view
    ax1 = fig.add_subplot(121, projection="3d")
    _plot_sites_and_atoms(ax1, pos_b, site_map, "Baseline", "blue")

    # Guided view
    ax2 = fig.add_subplot(122, projection="3d")
    _plot_sites_and_atoms(ax2, pos_g, site_map, "ESField Guided", "red")

    fig.suptitle(f"ESField Guided vs Baseline: {protein_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  3D scatter saved: {output_path}")


def _plot_sites_and_atoms(ax, pos, site_map, title, atom_color):
    pos_a = np.array(pos) if not isinstance(pos, np.ndarray) else pos
    # Plot sites as large markers
    for s in site_map["sites"]:
        c = s["center"]
        r = s["radius"]
        ax.scatter(*c, s=200, c="green", marker="o", alpha=0.7, edgecolors="darkgreen", linewidth=2)
        # Draw radius sphere (approximate as circle in 3D)
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = c[0] + r * np.outer(np.cos(u), np.sin(v))
        y = c[1] + r * np.outer(np.sin(u), np.sin(v))
        z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color="green", alpha=0.1, linewidth=0.3)

    # Plot atoms
    ax.scatter(pos_a[:, 0], pos_a[:, 1], pos_a[:, 2], c=atom_color, s=10, alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    # Equal aspect ratio
    _set_axes_equal(ax, pos_a)


def _set_axes_equal(ax, points):
    """Set 3D plot axes to equal scale."""
    x_limits = [points[:, 0].min(), points[:, 0].max()]
    y_limits = [points[:, 1].min(), points[:, 1].max()]
    z_limits = [points[:, 2].min(), points[:, 2].max()]
    max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]) / 2.0
    mid_x = (x_limits[0] + x_limits[1]) / 2
    mid_y = (y_limits[0] + y_limits[1]) / 2
    mid_z = (z_limits[0] + z_limits[1]) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_energy_histogram(energies_b, energies_g, output_path, protein_id):
    """Histogram of per-atom site energies: baseline vs guided."""
    fig, ax = plt.subplots(figsize=(10, 6))
    eb = energies_b.get("per_atom", [])
    eg = energies_g.get("per_atom", [])

    if eb and eg:
        bins = np.linspace(min(min(eb), min(eg)), max(max(eb), max(eg)), 50)
        ax.hist(eb, bins=bins, alpha=0.5, label=f"Baseline (μ={np.mean(eb):.3f})", color="blue", density=True)
        ax.hist(eg, bins=bins, alpha=0.5, label=f"ESField Guided (μ={np.mean(eg):.3f})", color="red", density=True)

    ax.set_xlabel("Site Energy (lower = more compatible)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Per-Atom Site Energy Distribution: {protein_id}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Energy histogram saved: {output_path}")


def plot_site_occupancy(occ_b, occ_g, output_path, protein_id):
    """Bar chart: site occupancy comparison per site."""
    site_ids = list(occ_b.keys())
    n = len(site_ids)
    if n == 0:
        return

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    vals_b = [occ_b[s]["occupancy"] for s in site_ids]
    vals_g = [occ_g[s]["occupancy"] for s in site_ids]

    bars1 = ax.bar(x - width/2, vals_b, width, label="Baseline", color="blue", alpha=0.7)
    bars2 = ax.bar(x + width/2, vals_g, width, label="ESField Guided", color="red", alpha=0.7)

    # Add site type labels
    for i, s in enumerate(site_ids):
        stype = occ_b[s]["site_type"]
        s_str = str(s)
        ax.text(i, max(vals_b[i], vals_g[i]) + 0.02, stype[:8], ha="center", fontsize=7, rotation=45)

    ax.set_xlabel("Site ID", fontsize=12)
    ax.set_ylabel("Occupancy (fraction)", fontsize=12)
    ax.set_title(f"Site Occupancy Comparison: {protein_id}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s)[:8] for s in site_ids], rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Site occupancy chart saved: {output_path}")


def plot_per_site_distance(dist_b, dist_g, output_path, protein_id):
    """Bar chart: mean distance to each site, baseline vs guided."""
    site_ids = list(dist_b.keys())
    n = len(site_ids)
    if n == 0:
        return

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    vals_b = [dist_b[s]["mean"] for s in site_ids]
    vals_g = [dist_g[s]["mean"] for s in site_ids]

    ax.bar(x - width/2, vals_b, width, label="Baseline", color="blue", alpha=0.7)
    ax.bar(x + width/2, vals_g, width, label="ESField Guided", color="red", alpha=0.7)

    ax.set_xlabel("Site ID", fontsize=12)
    ax.set_ylabel("Mean Distance to Site (Å)", fontsize=12)
    ax.set_title(f"Per-Site Mean Distance: {protein_id}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s)[:8] for s in site_ids], rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-site distance chart saved: {output_path}")


# ============================================================
# Main analysis
# ============================================================

def load_positions(result_path):
    """Load positions and atom types from a result file, handling multiple formats.

    First tries loading result_simple.pt, then falls back to original format.
    """
    # Try simple format first
    simple_path = Path(str(result_path).replace(".pt", "_simple.pt"))
    if simple_path.exists():
        r = torch.load(simple_path, map_location="cpu")
        return r["pos"].float().numpy(), r["v"].long().numpy()

    try:
        r = torch.load(result_path, map_location="cpu", weights_only=False)
    except Exception:
        # If unpickling fails (PAFlow format), bail out
        raise RuntimeError(f"Cannot load {result_path} — try converting to simple format first")

    if "pred_ligand_pos" in r:
        pos_list = []
        v_list = []
        for p in r["pred_ligand_pos"]:
            if isinstance(p, torch.Tensor):
                pos_list.append(p.float())
            else:
                pos_list.append(torch.from_numpy(np.array(p)).float())
        for v in r["pred_ligand_v"]:
            if isinstance(v, torch.Tensor):
                v_list.append(v.long())
            else:
                v_list.append(torch.from_numpy(np.array(v)).long())
        return torch.cat(pos_list, dim=0).numpy(), torch.cat(v_list, dim=0).numpy()
    else:
        pos = r["pos"].float()
        v = r["v"].long()
        if hasattr(pos, 'requires_grad') and pos.requires_grad:
            pos = pos.detach()
        return pos.numpy(), v.numpy()


def analyze_pocket(protein_id, baseline_pt, guided_pt, site_map_path, potential_ckpt, output_dir):
    """Full analysis for one pocket."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {protein_id}")
    print(f"{'='*60}")

    site_map = json.load(open(site_map_path))
    n_sites = len(site_map["sites"])

    # Load data
    pos_b, v_b = load_positions(baseline_pt)
    pos_g, v_g = load_positions(guided_pt)

    # Count receptor atoms
    n_atoms_b = len(pos_b)
    n_atoms_g = len(pos_g)
    print(f"Baseline: {n_atoms_b} atoms")
    print(f"Guided:  {n_atoms_g} atoms")
    print(f"Sites:   {n_sites}")

    # Load potential
    pot_ckpt = torch.load(potential_ckpt, map_location="cpu", weights_only=False)
    pot_cfg = pot_ckpt.get("config", {})
    pot_config = PotentialConfig(
        atom_embed_dim=pot_cfg.get("atom_embed_dim", 32),
        site_embed_dim=pot_cfg.get("site_embed_dim", 32),
        hidden_dim=pot_cfg.get("hidden_dim", 128),
        num_layers=pot_cfg.get("num_layers", 4),
    )
    potential = CompatibilityPotential(pot_config)
    potential.load_state_dict(pot_ckpt["model_state_dict"])
    potential.eval()

    # Site centers
    sites = [s["center"] for s in site_map["sites"]]

    # ========================================
    # Metric 1: Nearest site distance
    # ========================================
    nn_b = nearest_site_distance(pos_b, sites)
    nn_g = nearest_site_distance(pos_g, sites)
    print(f"\n--- Metric 1: Nearest Site Distance (Å) ---")
    print(f"  Baseline: mean={nn_b['mean']:.2f}, std={nn_b['std']:.2f}, max={nn_b['max']:.2f}")
    print(f"  Guided:   mean={nn_g['mean']:.2f}, std={nn_g['std']:.2f}, max={nn_g['max']:.2f}")
    if nn_b['mean'] > 0:
        delta_nn = (nn_g['mean'] - nn_b['mean']) / nn_b['mean'] * 100
        direction = "closer" if delta_nn < 0 else "farther"
        print(f"  Change:   {delta_nn:+.1f}% ({direction} to sites)")

    # ========================================
    # Metric 2: Site energy
    # ========================================
    energy_b = site_energy(pos_b, v_b, site_map, potential)
    energy_g = site_energy(pos_g, v_g, site_map, potential)
    print(f"\n--- Metric 2: Site Compatibility Energy ---")
    print(f"  Baseline: mean={energy_b['mean']:.4f}, std={energy_b['std']:.4f}, range=[{energy_b.get('min',0):.4f}, {energy_b.get('max',0):.4f}]")
    print(f"  Guided:   mean={energy_g['mean']:.4f}, std={energy_g['std']:.4f}, range=[{energy_g.get('min',0):.4f}, {energy_g.get('max',0):.4f}]")
    if abs(energy_b['mean']) > 1e-8:
        delta_e = (energy_g['mean'] - energy_b['mean']) / abs(energy_b['mean']) * 100
        better = "better" if energy_g['mean'] < energy_b['mean'] else "worse"
        print(f"  Change:   {delta_e:+.1f}% ({better} compatibility)")

    # ========================================
    # Metric 3: Site occupancy
    # ========================================
    occ_b = site_occupancy(pos_b, site_map)
    occ_g = site_occupancy(pos_g, site_map)
    print(f"\n--- Metric 3: Site Occupancy ---")
    for sid in occ_b:
        print(f"  Site {sid} ({occ_b[sid]['site_type']}): baseline={occ_b[sid]['occupancy']:.3f}, guided={occ_g[sid]['occupancy']:.3f}")

    # ========================================
    # Metric 4: Atom type distribution
    # ========================================
    at_dist_b = atom_type_distribution(v_b)
    at_dist_g = atom_type_distribution(v_g)
    print(f"\n--- Metric 4: Atom Type Distribution ---")
    print(f"  Baseline: {json.dumps(at_dist_b)}")
    print(f"  Guided:   {json.dumps(at_dist_g)}")

    # ========================================
    # Metric 5: Per-site distance
    # ========================================
    psd_b = per_site_distance(pos_b, site_map)
    psd_g = per_site_distance(pos_g, site_map)
    print(f"\n--- Metric 5: Per-Site Mean Distance (Å) ---")
    for sid in psd_b:
        print(f"  Site {sid}: baseline={psd_b[sid]['mean']:.2f}, guided={psd_g[sid]['mean']:.2f}")

    # ========================================
    # Visualizations
    # ========================================
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    plot_3d_scatter(pos_b, pos_g, site_map, vis_dir / f"{protein_id}_3d_scatter.png", protein_id)
    plot_energy_histogram(energy_b, energy_g, vis_dir / f"{protein_id}_energy_hist.png", protein_id)
    plot_site_occupancy(occ_b, occ_g, vis_dir / f"{protein_id}_site_occupancy.png", protein_id)
    plot_per_site_distance(psd_b, psd_g, vis_dir / f"{protein_id}_per_site_distance.png", protein_id)

    # Return summary for report
    return {
        "protein_id": protein_id,
        "n_atoms_baseline": n_atoms_b,
        "n_atoms_guided": n_atoms_g,
        "n_sites": n_sites,
        "nn_distance": {"baseline": nn_b, "guided": nn_g},
        "site_energy": {"baseline": {"mean": energy_b["mean"], "std": energy_b["std"]},
                        "guided": {"mean": energy_g["mean"], "std": energy_g["std"]}},  # noqa: E128
        "site_occupancy": {"baseline": occ_b, "guided": occ_g},
        "atom_type_dist": {"baseline": at_dist_b, "guided": at_dist_g},
        "per_site_distance": {"baseline": psd_b, "guided": psd_g},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets", nargs="*", default=None)
    parser.add_argument("--output-dir", default="experiments/analysis")
    parser.add_argument("--potential-ckpt", default="experiments/potential_training/train_gpu/compatibility_potential_epoch_0200.pt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results directory mapping
    results_base = ROOT / "experiments/matched_experiments"

    all_summaries = []

    pockets_to_analyze = args.pockets or ["FKB1A_HUMAN_2_108_0", "TIAM1_HUMAN_840_931_0"]

    for pid in pockets_to_analyze:
        baseline_pt = results_base / pid / "baseline" / "result.pt"
        guided_pt = results_base / pid / "guided" / "result.pt"
        site_map_path = ROOT / f"experiments/potential_training/site_maps/{pid}_site_map.json"

        # Fallback paths
        if not baseline_pt.exists():
            # Check old experiments
            for alt in [
                ROOT / f"experiments/esfield_paflow/{pid.split('_')[0]}/paflow_baseline/result.pt",
                ROOT / f"experiments/esfield_paflow/{pid}/paflow_baseline/result.pt",
            ]:
                if alt.exists():
                    baseline_pt = alt
                    break
        if not guided_pt.exists():
            for alt in [
                ROOT / f"experiments/esfield_paflow/{pid.split('_')[0]}/guided_v2/result.pt",
                ROOT / f"experiments/esfield_paflow/{pid}/guided_v2/result.pt",
            ]:
                if alt.exists():
                    guided_pt = alt
                    break

        print(f"\nChecking: {pid}")
        print(f"  baseline: {baseline_pt} ({'OK' if baseline_pt.exists() else 'MISSING'})")
        print(f"  guided:   {guided_pt} ({'OK' if guided_pt.exists() else 'MISSING'})")
        print(f"  site_map: {site_map_path} ({'OK' if site_map_path.exists() else 'MISSING'})")

        if baseline_pt.exists() and guided_pt.exists() and site_map_path.exists():
            summary = analyze_pocket(
                pid, baseline_pt, guided_pt, site_map_path,
                ROOT / args.potential_ckpt, output_dir
            )
            all_summaries.append(summary)

    # Save combined summary
    summary_path = output_dir / "all_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll metrics saved to {summary_path}")

    # Print final comparison table
    if all_summaries:
        print(f"\n{'='*80}")
        print(f"FINAL COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Metric':<30} {'Baseline':>15} {'Guided':>15} {'Change':>15}")
        print("-" * 80)
        for s in all_summaries:
            pid_short = s["protein_id"][:25]
            print(f"\n{pid_short}:")
            nn_b = s["nn_distance"]["baseline"]["mean"]
            nn_g = s["nn_distance"]["guided"]["mean"]
            e_b = s["site_energy"]["baseline"]["mean"]
            e_g = s["site_energy"]["guided"]["mean"]
            print(f"  {'NN Site Dist (Å)':<28} {nn_b:>15.3f} {nn_g:>15.3f} {((nn_g-nn_b)/nn_b*100):>+14.1f}%")
            print(f"  {'Site Energy':<28} {e_b:>15.4f} {e_g:>15.4f} {((e_g-e_b)/abs(e_b)*100):>+14.1f}%")


if __name__ == "__main__":
    main()
