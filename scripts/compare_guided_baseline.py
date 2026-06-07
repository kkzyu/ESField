#!/usr/bin/env python3
"""Compare ESField guided vs PAFlow baseline generation:
   - Mean distance from generated atoms to nearest site centers
   - Site energy (compatibility potential score) of generated molecules
   - RMSD between generated molecules

Usage:
  python scripts/compare_guided_baseline.py \
    --baseline-pt experiments/.../baseline/result.pt \
    --guided-pt experiments/.../guided/result.pt \
    --site-map experiments/.../site_map.json \
    --potential-ckpt experiments/.../checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAFLOW_ROOT = Path("/root/PAFlow-main")
sys.path.insert(0, str(PAFLOW_ROOT))
sys.path.insert(0, str(PAFLOW_ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np


def load_site_map(path):
    with open(path) as f:
        return json.load(f)


def nearest_site_distance(positions, site_centers):
    """Mean distance from each atom to its nearest site center."""
    pos = torch.tensor(positions) if not isinstance(positions, torch.Tensor) else positions
    centers = torch.tensor(site_centers) if not isinstance(site_centers, torch.Tensor) else site_centers
    if pos.ndim == 1:
        return float("nan")
    dists = torch.cdist(pos.float(), centers.float())
    min_dists = dists.min(dim=1).values
    return float(min_dists.mean())


def compute_site_energy(positions, atom_types, site_map, potential):
    """Compute mean compatibility energy per atom for the generated molecule."""
    centers = torch.tensor([s["center"] for s in site_map["sites"]], dtype=torch.float32)
    radii = torch.tensor([s["radius"] for s in site_map["sites"]], dtype=torch.float32)
    confs = torch.tensor([s["confidence"] for s in site_map["sites"]], dtype=torch.float32)
    stype_map = {"unknown": 0, "high_energy_water": 1, "stable_water": 2, "hydrophobic_cavity": 3}
    stypes = torch.tensor([stype_map.get(s["site_type"], 0) for s in site_map["sites"]], dtype=torch.long)

    n_atoms = len(positions)
    n_sites = len(centers)
    if n_sites == 0 or n_atoms == 0:
        return float("nan")

    pos = torch.tensor(positions, dtype=torch.float32) if not isinstance(positions, torch.Tensor) else positions.float()
    at = torch.tensor(atom_types, dtype=torch.long) if not isinstance(atom_types, torch.Tensor) else atom_types.long()
    # Clip to ESField vocab size (11 types)
    at = at.clamp(0, 10)

    # Tile for batched computation
    rel = pos[:, None, :] - centers[None, :, :]  # (A, S, 3)
    dist = torch.norm(rel, dim=-1).clamp_min(1e-8)  # (A, S)
    sigma = radii.clamp_min(1e-4)
    weight = torch.exp(-dist**2 / (2 * sigma[None, :]**2)) * confs[None, :]

    rel_flat = rel.reshape(-1, 3)
    dist_flat = dist.reshape(-1)
    st_flat = stypes.repeat(n_atoms)
    rad_flat = radii.repeat(n_atoms)
    conf_flat = confs.repeat(n_atoms)

    energies = []
    for at_idx in at.unique().tolist():
        mask = (at == at_idx)
        at_flat = torch.full((mask.sum().item() * n_sites,), at_idx, dtype=torch.long)
        e_flat = potential(
            at_flat,
            stypes.repeat(mask.sum().item()),
            rel[mask].reshape(-1, 3),
            dist[mask].reshape(-1),
            radii.repeat(mask.sum().item()),
            confs.repeat(mask.sum().item()),
        )
        e_grid = e_flat.reshape(mask.sum().item(), n_sites)
        energies.append((e_grid * weight[mask]).sum(dim=1))

    if energies:
        per_atom_energy = torch.cat(energies)
        return float(per_atom_energy.mean())
    return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pt", required=True)
    parser.add_argument("--guided-pt", required=True)
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--potential-ckpt", required=True)
    args = parser.parse_args()

    # Load data
    baseline = torch.load(args.baseline_pt, map_location="cpu", weights_only=False)
    guided = torch.load(args.guided_pt, map_location="cpu", weights_only=False)
    site_map = load_site_map(args.site_map)
    pot_ckpt = torch.load(args.potential_ckpt, map_location="cpu", weights_only=False)

    # Extract positions and atom types, handling different formats
    # PAFlow baseline format: {'pred_ligand_pos': list, 'pred_ligand_v': list}
    # ESField guided format: {'pos': tensor, 'v': tensor}
    if 'pred_ligand_pos' in baseline:
        b_pos = torch.cat([p.cpu() if isinstance(p, torch.Tensor) else torch.tensor(p) for p in baseline['pred_ligand_pos']])
        b_v = torch.cat([v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in baseline['pred_ligand_v']])
    else:
        b_pos = baseline['pos'].float()
        b_v = baseline['v'].long()

    if 'pred_ligand_pos' in guided:
        g_pos = torch.cat([p.cpu() if isinstance(p, torch.Tensor) else torch.tensor(p) for p in guided['pred_ligand_pos']])
        g_v = torch.cat([v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in guided['pred_ligand_v']])
    else:
        g_pos = guided['pos'].float()
        g_v = guided['v'].long()

    # Load potential
    from models.potential_network import CompatibilityPotential, PotentialConfig
    pot_cfg = pot_ckpt.get("config", {})
    pot_config = PotentialConfig(
        atom_embed_dim=pot_cfg.get("atom_embed_dim", 32),
        site_embed_dim=pot_cfg.get("site_embed_dim", 32),
        hidden_dim=pot_cfg.get("hidden_dim", 128),
        num_layers=pot_cfg.get("num_layers", 4),
        rbf_bins=pot_cfg.get("rbf_bins", 16),
        cutoff=pot_cfg.get("cutoff", 6.0),
        energy_clip=pot_cfg.get("energy_clip", 5.0),
    )
    potential = CompatibilityPotential(pot_config)
    potential.load_state_dict(pot_ckpt["model_state_dict"])
    potential.eval()

    sites = [s["center"] for s in site_map["sites"]]
    print(f"Site map: {len(sites)} sites")
    print(f"Baseline: {b_pos.shape[0]} atoms, {len(b_v)} types")
    print(f"Guided:  {g_pos.shape[0]} atoms, {len(g_v)} types")
    print()

    # Baseline metrics
    b_nn_dist = nearest_site_distance(b_pos, sites)
    b_energy = compute_site_energy(b_pos, b_v, site_map, potential)

    # Guided metrics
    g_nn_dist = nearest_site_distance(g_pos, sites)
    g_energy = compute_site_energy(g_pos, g_v, site_map, potential)

    print("=" * 50)
    print(f"{'Metric':<25} {'Baseline':>12} {'Guided':>12}")
    print("-" * 50)
    print(f"{'NN site dist (A)':<25} {b_nn_dist:>12.3f} {g_nn_dist:>12.3f}")
    print(f"{'Site energy':<25} {b_energy:>12.3f} {g_energy:>12.3f}")

    # Summary
    if g_nn_dist < b_nn_dist:
        improvement = (b_nn_dist - g_nn_dist) / b_nn_dist * 100
        print(f"\nESField guided atoms are {improvement:.1f}% closer to sites")
    if g_energy < b_energy:
        print(f"ESField guided has lower site energy (better compatibility)")


if __name__ == "__main__":
    main()
