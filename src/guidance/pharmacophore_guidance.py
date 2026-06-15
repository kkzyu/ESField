"""Pharmacophore-aware guidance — drops into existing DrugFlow pipeline.

Reuses SiteCompatibilityEnergy but swaps the HEW compatibility matrix for
the pharmacophore matrix.  Pharmacophore features (hbd, hba, hydrophobic,
aromatic, pos_ion, neg_ion) are treated as "sites" with a dedicated 6×11
compatibility matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from guidance.latent_guidance import (
    SiteCompatibilityEnergy,
    ATOM_TYPE_VOCAB, ATOM_TYPE_TO_IDX, N_ATOM_TYPES,
    PHARMACOPHORE_COMPAT_MATRIX, PHARMACOPHORE_TYPE_ORDER,
)

# Pharmacophore type → internal index
PHARM_TYPE_TO_IDX = {t: i for i, t in enumerate(PHARMACOPHORE_TYPE_ORDER)}


def build_pharmacophore_site_energy(
    pharm_site_map: dict[str, Any],
    sigma_distance: float = 3.0,
) -> SiteCompatibilityEnergy:
    """Build a SiteCompatibilityEnergy using pharmacophore compatibility matrix.

    Reads pharmacophore features from the site map JSON (produced by
    extract_pharmacophore.py).  Maps each pharmacophore type to the
    6-row pharmacophore compatibility matrix.

    Args:
        pharm_site_map: site map with "pharm_type" field on each site
        sigma_distance: Gaussian kernel width

    Returns:
        SiteCompatibilityEnergy with pharmacophore matrix and sites registered.
    """
    energy = SiteCompatibilityEnergy(
        sigma_distance=sigma_distance,
        compatibility_matrix=PHARMACOPHORE_COMPAT_MATRIX.clone(),
    )

    sites = pharm_site_map.get("sites", [])
    if not sites:
        return energy

    centers_list = []
    type_indices = []
    confidences = []

    for s in sites:
        ptype = s.get("pharm_type", "hydrophobic")
        centers_list.append(s["center"])
        type_indices.append(PHARM_TYPE_TO_IDX.get(ptype, 2))  # default: hydrophobic
        confidences.append(s.get("confidence", 1.0))

    centers = torch.tensor(centers_list, dtype=torch.float32)
    env_indices = torch.tensor(type_indices, dtype=torch.long)
    confs = torch.tensor(confidences, dtype=torch.float32)

    energy.register_sites(centers, env_indices, confs)
    return energy


class PharmacophoreGuideFn:
    """guide_log_prob callable for pharmacophore-aware generation.

    Computes E_pharm = -(1/τ) * log(Σ_i exp(τ * per_atom_score))
    where per_atom_score aggregates over pharmacophore sites.

    Supports:
      - Full gradient mode (per-atom gradient, all atoms)
      - CoM projection mode (KAG-style, gradient at centroid applied to all atoms)
    """

    def __init__(
        self,
        pharm_energy: SiteCompatibilityEnergy,
        lambda_guide: float = 1.0,
        guidance_start: float = 0.1,
        guidance_end: float = 0.90,
    ):
        self.pharm_energy = pharm_energy
        self.lambda_guide = lambda_guide
        self.esfield_lambda = lambda_guide  # for DrugFlow patched simulate()
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.grad_clip = 0.5

    def to(self, device: str) -> "PharmacophoreGuideFn":
        self.pharm_energy.to(device)
        return self

    def __call__(self, t_array, *, x, h, batch_mask, bonds=None, bond_types=None):
        t_val = float(t_array[0] if hasattr(t_array, "__len__") else t_array)
        if t_val < self.guidance_start or t_val > self.guidance_end:
            return torch.tensor(0.0, device=x.device)
        if self.pharm_energy.n_sites == 0:
            return torch.tensor(0.0, device=x.device)

        h_sum = h.sum(dim=-1)
        if (h >= 0).all() and (h <= 1).all() and torch.allclose(h_sum, torch.ones_like(h_sum), atol=0.01):
            atom_probs = h
        else:
            atom_probs = F.softmax(h, dim=-1)

        e_pharm = self.pharm_energy(x, atom_type_probs=atom_probs)
        return -self.lambda_guide * e_pharm


def load_pharm_site_map(path: str | Path) -> dict:
    """Load a pharmacophore site map JSON."""
    with open(path) as f:
        return json.load(f)
