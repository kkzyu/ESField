"""Two-stage hierarchical generation for v7 latent guidance.

Phase 1 — OCCUPY:
  Generate a small fragment (3–5 atoms) with strong site-compatibility
  guidance.  The goal is to place at least one compatible atom within
  2.5 Å of a candidate HEW site.  This "anchor fragment" seeds the
  topology needed for site utilization.

Phase 2 — CONNECT:
  Continue generation from the Phase 1 anchor atoms to build a full
  drug-like molecule.  The anchor atoms are softly restrained to their
  Phase 1 positions (harmonic potential), while the remaining atoms
  grow under weaker guidance toward other candidate sites.

Key design principle (from v6-D.2 NO-GO diagnosis):
  Candidate HEW utilization requires TOPOLOGY-LEVEL control (adding
  atoms, changing types, growing substituents), not just coordinate
  nudging.  The two-stage approach achieves this by (a) explicitly
  growing a fragment at the target site first, then (b) connecting
  it into a full molecule.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from guidance.latent_guidance import (
    COMPAT_MATRIX,
    HEW_ENV_HYDROPHOBIC,
    HEW_ENV_MIXED,
    HEW_ENV_POLAR_UNSATISFIED,
    HEW_ENV_BURIED,
    ATOM_TYPE_VOCAB,
    ATOM_TYPE_TO_IDX,
    N_ATOM_TYPES,
    SiteCompatibilityEnergy,
    TypeGuidanceBias,
    hew_env_to_idx,
    apply_latent_guidance,
    build_site_energy_from_map,
    classify_hew_environment,
    harmonic_restraint_energy,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler, CompositeScheduler
from guidance.hard_fix import HardFixCallback, patch_drugflow_hardfix


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Phase1Config:
    """Configuration for Phase 1 (Occupy)."""

    n_init_atoms: int = 4
    """Number of atoms in the initial small fragment."""

    max_steps: int = 100
    """Maximum ODE integration steps for Phase 1."""

    success_distance: float = 2.5
    """An atom within this distance (Å) of a HEW site with compat >= min
    marks a successful occupation."""

    min_compatibility: float = 0.3
    """Minimum compatibility score (from the matrix) for a successful
    occupation.  Values >= 0.3 mean at least weakly compatible."""

    attempts: int = 3
    """Number of retry attempts if Phase 1 fails to occupy any site."""

    lambda_early: float = 5.0
    """Guidance strength λ for Phase 1 (strong, to drive occupation)."""

    kts_alpha0: float = 0.01
    """KTS early boost strength for Phase 1."""

    kts_beta0: float = 0.01
    """KTS late damping strength for Phase 1."""

    guidance_start: float = 0.05
    """Start guidance early in Phase 1 (noisy regime → topology changes)."""

    guidance_end: float = 0.95
    """Keep guidance active throughout Phase 1."""

    grad_clip: float = 0.5
    """Gradient clipping for Phase 1 (lower = more stable)."""

    sigma_distance: float = 3.0
    """Gaussian kernel width for site-compatibility energy."""

    # Post-Phase1: which atoms to keep as anchors
    anchor_selection: str = "best_per_site"
    """Strategy for selecting anchor atoms from Phase 1 output:
    - "best_per_site": best compatible atom for each occupied site
    - "all_compatible": all atoms with compat >= min_compatibility and d < success_distance
    - "nearest_compatible": nearest compatible atom overall
    """


@dataclass
class Phase2Config:
    """Configuration for Phase 2 (Connect)."""

    # ── Anchor fixation mode ──
    anchor_fix_mode: str = "hard"
    """Anchor fixation strategy:
    - "hard": v7.1 default — hard-overwrite coordinates every step
    - "hard_locked": ★ Fully locked — anchors removed before ODE,
      re-inserted after. Zero KPE from anchors. Fair baseline.
    - "annealing": v7.1a — hard-fix for fix_fraction of steps, then
      harmonic restraint decaying to zero
    - "soft": harmonic restraint only (no hard-overwrite), k = restraint_force
    - "kinematic": ★ SV-Flow fusion — kinematic decoupling + CoM-level
      site-attraction guidance.  Preserves v_int (internal conformational
      velocity), mathematically guaranteeing zero strain on chemical bonds
      while softly guiding anchor CoM toward HEW sites.
    """

    fix_atoms: bool = True
    """Whether to fix anchor atom positions.  If False, use only harmonic
    restraints (allowing small fluctuations)."""

    restraint_force: float = 10.0
    """Harmonic force constant for anchor atom restraints (kcal/mol/Å²
    equivalent).  Higher = tighter restraint."""

    # ── Annealing parameters (used when anchor_fix_mode == "annealing") ──
    annealing_fix_fraction: float = 0.7
    """Fraction of total Phase 2 steps to hard-fix before annealing."""

    annealing_restraint_start: float = 10.0
    """Initial harmonic force constant at start of annealing phase."""

    annealing_restraint_end: float = 0.0
    """Final harmonic force constant at end of annealing phase."""

    annealing_ramp: str = "linear"
    """Decay schedule: "linear" or "exponential"."""

    # ── Kinematic parameters (used when anchor_fix_mode == "kinematic") ──
    kinematic_lambda_max: float = 1.0
    """Maximum guidance strength λ_max for kinematic anchor guidance.
    Controls how strongly the anchor CoM is pulled toward HEW sites.
    Higher = stronger attraction (0.1–2.0).  Default 1.0."""

    kinematic_profile: str = "quadratic"
    """λ(t) decay profile: "quadratic", "constant", "late_onset", "linear".
    "quadratic" (default) smoothly decays guidance toward the end,
    allowing conformational fine-tuning."""

    kinematic_t_on: float = 0.5
    """Onset time for "late_onset" profile.  Only used with that profile."""

    kinematic_grad_clip: float = 0.5
    """Maximum per-step anchor CoM correction magnitude (Å).  Prevents
    large jumps while maintaining smooth guidance."""

    kinematic_track_kpe: bool = True
    """Whether to collect KPE diagnostics during kinematic guidance.
    Enables the cross-method KPE comparison experiment (E2)."""

    type_bias_strength: float = 0.0
    """Strength of type-preservation bias for anchor atoms.
    Uses cross-entropy loss to penalize type changes.  0 = disabled.
    Recommended: 0.1-1.0 when hard_fix doesn't cover types."""

    max_total_steps: int = 200
    """Maximum total ODE steps for Phase 2."""

    lambda_late: float = 0.1
    """Guidance strength for Phase 2 (weaker — don't over-constrain)."""

    guidance_start: float = 0.1
    """When to start guidance in Phase 2."""

    guidance_end: float = 0.90
    """When to end guidance in Phase 2."""

    grad_clip: float = 0.3
    """Gradient clipping for Phase 2."""

    kts_alpha0: float = 0.005
    """KTS boost for Phase 2 (milder than Phase 1)."""

    kts_beta0: float = 0.01
    """KTS damping for Phase 2."""


@dataclass
class TwoStageConfig:
    """Full v7 two-stage generation configuration."""

    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)

    # Post-processing
    minimize: bool = True
    force_field: str = "MMFF94"

    # Logging
    verbose: bool = True


# ---------------------------------------------------------------------------
# Anchor atom representation
# ---------------------------------------------------------------------------


@dataclass
class AnchorAtoms:
    """Set of fixed atoms from Phase 1 that seed Phase 2."""

    positions: torch.Tensor  # [n_anchors, 3]  coordinates
    type_indices: torch.Tensor  # [n_anchors] atom type indices
    type_probs: torch.Tensor  # [n_anchors, n_atom_types] soft type probs
    occupied_sites: list[int]  # which HEW sites are occupied
    compat_scores: list[float]  # compatibility score for each anchor
    distances: list[float]  # distance to nearest site for each anchor

    @property
    def n_anchors(self) -> int:
        return self.positions.shape[0]

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "n_anchors": self.n_anchors,
            "positions": self.positions.cpu().tolist(),
            "type_indices": self.type_indices.cpu().tolist(),
            "type_probs": self.type_probs.cpu().tolist(),
            "occupied_sites": self.occupied_sites,
            "compat_scores": self.compat_scores,
            "distances": self.distances,
        }


# ---------------------------------------------------------------------------
# Anchor Type Selector — suggests optimal atom types for each HEW environment
# ---------------------------------------------------------------------------


def suggest_anchor_types(
    site_env: str,
    top_k: int = 4,
) -> list[str]:
    """Suggest the best anchor atom types for a given HEW environment.

    Based on the hard-coded compatibility matrix.  Types are ranked by
    their compatibility score with the site environment.

    Args:
        site_env: one of "hydrophobic", "polar_unsatisfied", "mixed", "buried"
        top_k: number of types to return

    Returns:
        List of atom type names (from ATOM_TYPE_VOCAB), best first.

    Examples:
        >>> suggest_anchor_types("hydrophobic", 3)
        ['C_sp3', 'C_aromatic', 'halogen']
        >>> suggest_anchor_types("polar_unsatisfied", 3)
        ['N_donor', 'N_acceptor', 'O_acceptor']
    """
    env_idx = hew_env_to_idx(site_env)
    compat_row = COMPAT_MATRIX[env_idx]

    # Score each type, exclude "unknown" and "charged" from suggestions
    scored = []
    for type_idx, type_name in enumerate(ATOM_TYPE_VOCAB):
        if type_name in ("unknown", "charged"):
            continue
        score = compat_row[type_idx].item()
        scored.append((type_name, score))

    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored[:top_k]]


class AnchorTypeSelector:
    """Selects anchor atom types for Phase 1 based on HEW site environments.

    Instead of relying on DrugFlow to randomly generate the right atom type,
    this class suggests specific types for each site environment, trying
    the most compatible types first.

    Usage:
        selector = AnchorTypeSelector(site_map, strategy="suggested")
        suggested_types = selector.get_types_for_attempt(attempt_idx)
        # Pass suggested_types to Phase 1 guide or use to bias generation.
    """

    def __init__(
        self,
        site_map: dict,
        strategy: str = "suggested",
        max_attempts_per_type: int = 2,
    ):
        """
        Args:
            site_map: site map dict
            strategy: "suggested" (use compatibility-ranked types) or
                      "random" (use all types equally)
            max_attempts_per_type: how many Phase 1 attempts to try
                                   with each suggested type before
                                   moving to the next
        """
        self.strategy = strategy
        self.max_attempts_per_type = max_attempts_per_type

        # Collect all HEW environments present
        hew_sites = [s for s in site_map["sites"]
                     if s.get("site_type") == "high_energy_water"]
        env_counts: dict[str, int] = {}
        for s in hew_sites:
            env = classify_hew_environment(s)
            env_counts[env] = env_counts.get(env, 0) + 1

        # Build the suggested type priority queue
        self._type_queue: list[str] = []
        if strategy == "suggested":
            # Prioritize types that work for the MOST COMMON environments
            # For each environment, add its top types (deduplicated, in order)
            seen: set[str] = set()
            # Sort environments by frequency (most common first)
            sorted_envs = sorted(env_counts.keys(), key=lambda e: -env_counts[e])
            for env in sorted_envs:
                suggested = suggest_anchor_types(env, top_k=4)
                for t in suggested:
                    if t not in seen:
                        self._type_queue.append(t)
                        seen.add(t)
        else:
            # Random: use all compatible types
            self._type_queue = [
                t for t in ATOM_TYPE_VOCAB
                if t not in ("unknown", "charged", "B", "P")
            ]

        self._env_counts = env_counts
        self._current_type_idx = 0

    @property
    def env_distribution(self) -> dict[str, int]:
        """Distribution of HEW environments in the site map."""
        return dict(self._env_counts)

    def get_types_for_attempt(self, attempt_idx: int, n_types: int = 3) -> list[str]:
        """Get suggested anchor types for a given Phase 1 attempt.

        Args:
            attempt_idx: 0-based attempt number
            n_types: number of types to return

        Returns:
            List of atom type names to prioritize in this attempt.
        """
        if self.strategy == "random":
            import random
            return random.sample(self._type_queue, min(n_types, len(self._type_queue)))

        # "suggested": cycle through type queue with max_attempts_per_type
        type_idx = attempt_idx // self.max_attempts_per_type
        type_idx = type_idx % len(self._type_queue)

        # Return the current type and the next few as fallbacks
        result = []
        for offset in range(n_types):
            idx = (type_idx + offset) % len(self._type_queue)
            result.append(self._type_queue[idx])
        return result

    def get_all_suggested_types(self) -> list[str]:
        """Return all suggested types in priority order."""
        return list(self._type_queue)


# ---------------------------------------------------------------------------
# Phase 1: Occupy
# ---------------------------------------------------------------------------


def _compute_diagnostics(
    x: torch.Tensor,
    h: torch.Tensor,
    site_energy: SiteCompatibilityEnergy,
    success_distance: float,
    min_compatibility: float,
) -> dict[str, Any]:
    """Compute Phase 1 success diagnostics.

    Checks whether any atom is within success_distance of a HEW site
    AND has compatibility score >= min_compatibility.
    """
    n_atoms = x.shape[0]
    n_sites = site_energy.n_sites

    if n_sites == 0:
        return {
            "success": False,
            "n_sites": 0,
            "best_distance": float("inf"),
            "best_compat": 0.0,
            "n_occupied_sites": 0,
            "per_site_best": [],
        }

    # Pairwise distances
    centers = site_energy._site_centers
    env_indices = site_energy._site_env_indices
    rel = x[:, None, :] - centers[None, :, :]
    dist = torch.linalg.norm(rel, dim=-1)  # [n_atoms, n_sites]

    # Compatibility: handle both logits and pre-softmaxed probabilities.
    # If h values are all in [0,1] and sum to ~1, treat as probabilities;
    # otherwise apply softmax (h is raw logits, as in DrugFlow).
    h_sum = h.sum(dim=-1)
    if (h >= 0).all() and (h <= 1).all() and torch.allclose(h_sum, torch.ones_like(h_sum), atol=0.01):
        # Already probability distributions
        atom_probs = h
    else:
        atom_probs = F.softmax(h, dim=-1)

    n_types = min(atom_probs.shape[-1], N_ATOM_TYPES)
    probs = atom_probs[:, :n_types]
    env_compat = site_energy.compatibility_matrix[env_indices, :][:, :n_types]
    compat = torch.matmul(probs, env_compat.T)  # [n_atoms, n_sites]

    # Hard type assignment for diagnostics
    hard_types = atom_probs.argmax(dim=-1)

    # Check each site
    occupied_sites = []
    per_site_best = []
    best_overall_dist = float("inf")
    best_overall_compat = 0.0

    for j in range(n_sites):
        site_dist = dist[:, j]
        site_compat = compat[:, j]
        min_d, best_i = site_dist.min(dim=0)
        best_c = site_compat[best_i].item()
        best_type = ATOM_TYPE_VOCAB[hard_types[best_i].item()]

        per_site_best.append({
            "site_idx": j,
            "min_distance": min_d.item(),
            "best_compat": best_c,
            "best_atom_idx": best_i.item(),
            "best_atom_type": best_type,
            "occupied": (min_d.item() <= success_distance and best_c >= min_compatibility),
        })

        if min_d.item() <= success_distance and best_c >= min_compatibility:
            occupied_sites.append(j)

        if min_d.item() < best_overall_dist:
            best_overall_dist = min_d.item()
        if best_c > best_overall_compat:
            best_overall_compat = best_c

    return {
        "success": len(occupied_sites) > 0,
        "n_sites": n_sites,
        "n_occupied_sites": len(occupied_sites),
        "occupied_site_indices": occupied_sites,
        "best_distance": best_overall_dist,
        "best_compat": best_overall_compat,
        "per_site_best": per_site_best,
    }


def _extract_anchors(
    x: torch.Tensor,
    h: torch.Tensor,
    site_energy: SiteCompatibilityEnergy,
    diagnostics: dict,
    config: Phase1Config,
) -> AnchorAtoms | None:
    """Extract anchor atoms from successful Phase 1 output.

    Returns None if no compatible atoms are found near sites.
    """
    if not diagnostics["success"]:
        return None

    n_atoms = x.shape[0]
    centers = site_energy._site_centers
    env_indices = site_energy._site_env_indices

    # Auto-detect: logits vs already-softmaxed probabilities
    h_sum3 = h.sum(dim=-1)
    if (h >= 0).all() and (h <= 1).all() and torch.allclose(h_sum3, torch.ones_like(h_sum3), atol=0.01):
        atom_probs = h
    else:
        atom_probs = F.softmax(h, dim=-1)
    n_types = min(atom_probs.shape[-1], N_ATOM_TYPES)
    probs = atom_probs[:, :n_types]
    env_compat = site_energy.compatibility_matrix[env_indices, :][:, :n_types]
    compat = torch.matmul(probs, env_compat.T)

    rel = x[:, None, :] - centers[None, :, :]
    dist = torch.linalg.norm(rel, dim=-1)

    hard_types = atom_probs.argmax(dim=-1)

    occupied_sites = diagnostics["occupied_site_indices"]

    if config.anchor_selection == "best_per_site":
        # Take the best compatible atom for each occupied site
        anchor_indices = set()
        for j in occupied_sites:
            site_dist = dist[:, j]
            site_compat = compat[:, j]
            # Find compatible atoms within success distance
            valid_mask = (
                (site_dist <= config.success_distance)
                & (site_compat >= config.min_compatibility)
            )
            if valid_mask.any():
                # Pick the one with highest compat * gaussian weight
                sigma2 = 2.0 * config.success_distance ** 2
                scores = site_compat * torch.exp(-(site_dist ** 2) / sigma2)
                scores[~valid_mask] = -float("inf")
                best_i = scores.argmax().item()
                anchor_indices.add(best_i)

    elif config.anchor_selection == "all_compatible":
        # Take all compatible atoms within distance threshold
        anchor_indices = set()
        for j in range(site_energy.n_sites):
            site_dist = dist[:, j]
            site_compat = compat[:, j]
            valid_mask = (
                (site_dist <= config.success_distance)
                & (site_compat >= config.min_compatibility)
            )
            for i in valid_mask.nonzero(as_tuple=True)[0]:
                anchor_indices.add(i.item())

    elif config.anchor_selection == "nearest_compatible":
        # Single nearest compatible atom
        valid_mask = (dist.min(dim=1)[0] <= config.success_distance) & (
            compat.max(dim=1)[0] >= config.min_compatibility
        )
        if valid_mask.any():
            anchor_indices = {valid_mask.nonzero(as_tuple=True)[0][0].item()}
        else:
            anchor_indices = set()
    else:
        anchor_indices = set()

    if not anchor_indices:
        return None

    anchor_list = sorted(anchor_indices)

    # Build anchor positions and types
    positions = x[anchor_list].detach().clone()
    type_indices = hard_types[anchor_list]
    type_probs_values = atom_probs[anchor_list].detach().clone()

    # Find which sites each anchor occupies.
    # For each anchor, record the CLOSEST site that it actually occupies
    # (d ≤ success_distance AND compat ≥ min_compatibility), not just the
    # site with highest compatibility (which could be far away).
    anchor_occupied_sites = []
    anchor_compat_scores = []
    anchor_distances = []
    for i in anchor_list:
        site_dist = dist[i]
        site_compat = compat[i]
        # Find the closest site that meets occupancy criteria
        occupied_mask = (site_dist <= config.success_distance) & (
            site_compat >= config.min_compatibility
        )
        if occupied_mask.any():
            # Among occupied sites, pick the closest one
            masked_dist = site_dist.clone()
            masked_dist[~occupied_mask] = float("inf")
            best_j = masked_dist.argmin().item()
            anchor_occupied_sites.append(best_j)
            anchor_compat_scores.append(site_compat[best_j].item())
            anchor_distances.append(site_dist[best_j].item())
        else:
            # Fallback: use closest site by distance
            best_j = site_dist.argmin().item()
            anchor_occupied_sites.append(best_j)
            anchor_compat_scores.append(site_compat[best_j].item())
            anchor_distances.append(site_dist[best_j].item())

    return AnchorAtoms(
        positions=positions,
        type_indices=type_indices,
        type_probs=type_probs_values,
        occupied_sites=anchor_occupied_sites,
        compat_scores=anchor_compat_scores,
        distances=anchor_distances,
    )


# ---------------------------------------------------------------------------
# Phase 2 guide function (for use with DrugFlow's guide_log_prob)
# ---------------------------------------------------------------------------


class TwoStageGuideFn:
    """Composite guide_log_prob callable for Phase 2.

    Combines:
      1. Site-compatibility energy for remaining HEW sites
      2. Harmonic restraint for anchor atoms from Phase 1
      3. Optional type biasing toward compatible types

    This class is designed to be passed as `guide_log_prob` to
    DrugFlow's model.sample() method.
    """

    def __init__(
        self,
        site_energy: SiteCompatibilityEnergy,
        anchors: AnchorAtoms,
        config: Phase2Config,
        kts: KTSScheduler | None = None,
        type_bias_strength: float = 0.0,
    ):
        self.site_energy = site_energy
        self.anchors = anchors
        self.config = config
        self.kts = kts or KTSScheduler(
            alpha0=config.kts_alpha0,
            beta0=config.kts_beta0,
        )
        self.type_bias_strength = type_bias_strength

        # Internal state
        self.guidance_start = config.guidance_start
        self.guidance_end = config.guidance_end
        self.esfield_lambda = config.lambda_late
        self.grad_clip = config.grad_clip
        self.restraint_force = config.restraint_force

        # Anchor mask will be set during sampling based on atom count
        self._anchor_mask: torch.Tensor | None = None
        self._anchor_positions: torch.Tensor | None = None

    def to(self, device: str) -> "TwoStageGuideFn":
        """Move tensors to device."""
        self.site_energy.to(device)
        self.anchors.positions = self.anchors.positions.to(device)
        self.anchors.type_probs = self.anchors.type_probs.to(device)
        self.anchors.type_indices = self.anchors.type_indices.to(device)
        return self

    def set_anchor_indices(self, atom_indices: list[int], total_atoms: int):
        """Set which atom indices in the full molecule are anchors.

        Args:
            atom_indices: indices (0-based) of anchor atoms in the full molecule
            total_atoms: total number of atoms in the full molecule
        """
        mask = torch.zeros(total_atoms, dtype=torch.bool)
        mask[atom_indices] = True
        self._anchor_mask = mask
        self._anchor_positions = torch.zeros(total_atoms, 3)
        for i, idx in enumerate(atom_indices):
            self._anchor_positions[idx] = self.anchors.positions[i]

    def __call__(
        self,
        t_array,
        *,
        x,
        h,
        batch_mask,
        bonds=None,
        bond_types=None,
    ) -> torch.Tensor:
        """Compute guidance energy for Phase 2.

        Returns negative energy (so minimizing energy = maximizing log_prob).
        Compatible with DrugFlow's guide_log_prob interface.
        """
        n_atoms = x.shape[0]
        device = x.device
        t_val = float(t_array[0] if hasattr(t_array, "__len__") else t_array)

        # Time gating
        if t_val < self.guidance_start or t_val > self.guidance_end:
            return torch.tensor(0.0, device=device)

        # KTS scaling
        eta = self.kts(t_val)
        lam_eff = self.esfield_lambda * eta
        if lam_eff <= 0:
            return torch.tensor(0.0, device=device)

        # Auto-detect whether h is logits or probabilities
        h_sum4 = h.sum(dim=-1)
        if (h >= 0).all() and (h <= 1).all() and torch.allclose(h_sum4, torch.ones_like(h_sum4), atol=0.01):
            atom_probs = h
        else:
            atom_probs = F.softmax(h, dim=-1)

        total_energy = torch.tensor(0.0, device=device)

        # 1. Site-compatibility energy (for ALL atoms toward ALL sites)
        if self.site_energy.n_sites > 0:
            e_site = self.site_energy(x, atom_type_probs=atom_probs)
            total_energy = total_energy + lam_eff * e_site

        # 2. Harmonic restraint for anchor atoms
        if (
            self._anchor_mask is not None
            and self._anchor_mask.any()
            and self.config.restraint_force > 0
        ):
            # Move anchor positions to device if needed
            if self._anchor_positions.device != device:
                self._anchor_positions = self._anchor_positions.to(device)
                self._anchor_mask = self._anchor_mask.to(device)

            # Adapt mask to actual n_atoms (DrugFlow may generate different
            # numbers of atoms than the requested num_nodes).
            if self._anchor_mask.shape[0] != n_atoms:
                mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
                target = torch.zeros(n_atoms, 3, device=device)
                n_copy = min(self._anchor_mask.shape[0], n_atoms)
                mask[:n_copy] = self._anchor_mask[:n_copy]
                target[:n_copy] = self._anchor_positions[:n_copy]
            else:
                mask = self._anchor_mask
                target = self._anchor_positions

            e_restraint = harmonic_restraint_energy(
                x, mask, target, force_constant=self.config.restraint_force
            )
            total_energy = total_energy + e_restraint

        # 3. Type bias: penalize anchor atoms that deviate from Phase 1 types.
        # Uses cross-entropy: -log(softmax(h_anchor)[target_type])
        if (
            self.type_bias_strength > 0
            and self._anchor_mask is not None
            and self._anchor_mask.any()
            and self.anchors.type_indices is not None
        ):
            if self.anchors.type_indices.device != device:
                self.anchors.type_indices = self.anchors.type_indices.to(device)

            # Get anchor atom indices that are within the current n_atoms
            n_anchors_in_bounds = min(
                self.anchors.type_indices.shape[0],
                n_atoms,
                self._anchor_mask.shape[0],
            )
            if n_anchors_in_bounds > 0:
                anchor_h = h[:n_anchors_in_bounds]  # [n_anchors, n_features]
                target_types = self.anchors.type_indices[:n_anchors_in_bounds]

                # Cross-entropy: -log(softmax(h)[target])
                # = -(h[target] - logsumexp(h))
                log_probs = F.log_softmax(anchor_h, dim=-1)
                nll = -log_probs[range(n_anchors_in_bounds), target_types].mean()
                total_energy = total_energy + self.type_bias_strength * nll

        # Return -energy (DrugFlow convention: maximize log_prob = minimize energy)
        return -total_energy


# ---------------------------------------------------------------------------
# merge_anchors_to_initial_molecule — prepare Phase 2 initial state from Phase 1
# ---------------------------------------------------------------------------


def merge_anchors_to_initial_molecule(
    phase1_fragment_mol,
    anchor_list: "AnchorAtoms",
) -> dict[str, Any]:
    """Prepare Phase 2 initial molecule from Phase 1 fragment and anchors.

    The Phase 1 fragment (all atoms and bonds) becomes the seed for Phase 2.
    Anchor atoms are a subset of the fragment that occupy HEW sites.

    Args:
        phase1_fragment_mol: RDKit Mol from Phase 1 (full fragment, all atoms+bonds)
        anchor_list: AnchorAtoms identifying which fragment atoms are anchors

    Returns:
        dict with:
          - "fragment_mol": the Phase 1 RDKit Mol (same as input)
          - "anchor_positions": [n_anchors, 3] tensor
          - "anchor_type_indices": [n_anchors] tensor
          - "n_fragment_atoms": int
          - "n_anchors": int
    """
    return {
        "fragment_mol": phase1_fragment_mol,
        "anchor_positions": anchor_list.positions.clone(),
        "anchor_type_indices": anchor_list.type_indices.clone(),
        "n_fragment_atoms": phase1_fragment_mol.GetNumAtoms() if phase1_fragment_mol is not None else 0,
        "n_anchors": anchor_list.n_anchors,
    }


# ---------------------------------------------------------------------------
# TwoStageGenerator — main orchestrator
# ---------------------------------------------------------------------------


class TwoStageGenerator:
    """Orchestrate v7 two-stage generation.

    Usage pattern:
        gen = TwoStageGenerator(config, model, site_map)
        gen.to("cuda:0")

        # Phase 1: generate anchor fragment
        anchors = gen.phase1_occupy(protein_data, ref_size)

        if anchors is not None:
            # Phase 2: grow full molecule
            mol = gen.phase2_connect(protein_data, anchors, full_size)
    """

    def __init__(
        self,
        config: TwoStageConfig,
        model,  # DrugFlow model
        site_map: dict | str | Path,
    ):
        self.config = config
        self.model = model

        # Load site map
        self._site_map = (
            site_map
            if isinstance(site_map, dict)
            else json.loads(Path(site_map).read_text())
        )

        # Build site energy for Phase 1 and Phase 2
        # Phase 1 uses all HEW environments for broad capture
        self._site_energy_p1 = build_site_energy_from_map(
            self._site_map,
            sigma_distance=config.phase1.sigma_distance,
            enabled_envs=(
                HEW_ENV_HYDROPHOBIC,
                HEW_ENV_POLAR_UNSATISFIED,
                HEW_ENV_MIXED,
            ),
        )

        # Phase 2 can use a different filter if needed
        self._site_energy_p2 = build_site_energy_from_map(
            self._site_map,
            sigma_distance=config.phase1.sigma_distance,
            enabled_envs=(
                HEW_ENV_HYDROPHOBIC,
                HEW_ENV_POLAR_UNSATISFIED,
                HEW_ENV_MIXED,
            ),
        )

        # KTS schedulers
        self._kts_p1 = KTSScheduler(
            alpha0=config.phase1.kts_alpha0,
            beta0=config.phase1.kts_beta0,
        )
        self._kts_p2 = KTSScheduler(
            alpha0=config.phase2.kts_alpha0,
            beta0=config.phase2.kts_beta0,
        )

        # Logging
        self._phase1_log: list[dict] = []
        self._phase2_log: list[dict] = []

        # Phase 1 fragment (stored when Phase 1 succeeds)
        self._phase1_fragment_mol = None  # RDKit Mol from Phase 1
        self._anchors: AnchorAtoms | None = None

    @property
    def n_hew_sites(self) -> int:
        return self._site_energy_p1.n_sites

    def to(self, device: str) -> "TwoStageGenerator":
        self._site_energy_p1.to(device)
        self._site_energy_p2.to(device)
        return self

    # ------------------------------------------------------------------
    # Phase 1: Occupy
    # ------------------------------------------------------------------

    def phase1_occupy(
        self,
        protein_data: dict,
        ref_mol_size: int | None = None,
        *,
        n_samples_per_attempt: int = 3,
        timesteps: int = 50,
        device: str = "cuda:0",
    ) -> AnchorAtoms | None:
        """Run Phase 1: generate a small fragment that occupies a HEW site.

        Args:
            protein_data: DrugFlow protein batch dict (pre-processed)
            ref_mol_size: reference ligand size (used for context only)
            n_samples_per_attempt: molecules to generate per attempt
            timesteps: ODE integration steps for Phase 1
            device: torch device

        Returns:
            AnchorAtoms if successful, None if all attempts failed.
            The fragment RDKit Mol is stored in self._phase1_fragment_mol.
        """
        cfg = self.config.phase1
        self._site_energy_p1.to(device)
        self._phase1_log = []

        for attempt in range(cfg.attempts):
            if self.config.verbose:
                print(
                    f"  Phase 1 attempt {attempt + 1}/{cfg.attempts} "
                    f"(n_atoms={cfg.n_init_atoms}, λ={cfg.lambda_early})"
                )

            # Build guide function for Phase 1
            guide_fn = _Phase1GuideFn(
                site_energy=self._site_energy_p1,
                lambda_guide=cfg.lambda_early,
                guidance_start=cfg.guidance_start,
                guidance_end=cfg.guidance_end,
                grad_clip=cfg.grad_clip,
                kts=self._kts_p1,
            ).to(device)

            # Generate small fragments
            t0 = time.time()
            with torch.no_grad():
                rdmols, trajectories, _ = self.model.sample(
                    protein_data,
                    n_samples=n_samples_per_attempt,
                    timesteps=timesteps,
                    num_nodes=cfg.n_init_atoms,
                    guide_log_prob=guide_fn,
                )
            elapsed = time.time() - t0

            # Check each generated molecule for site occupancy
            for mol_idx, mol in enumerate(rdmols):
                if mol is None:
                    continue

                # Get coordinates and atom features from the final state
                # Note: DrugFlow's sample() returns RDKit mols directly.
                # We need the intermediate tensor states for diagnostics.
                # For MVP, we extract coordinates from the RDKit mol and
                # map atom types.
                x, h = _tensors_from_rdmol(
                    mol, device=device
                )

                if x is None or x.shape[0] == 0:
                    continue

                diag = _compute_diagnostics(
                    x, h, self._site_energy_p1,
                    cfg.success_distance, cfg.min_compatibility,
                )

                attempt_log = {
                    "attempt": attempt + 1,
                    "mol_idx": mol_idx,
                    "success": diag["success"],
                    "n_occupied_sites": diag["n_occupied_sites"],
                    "best_distance": diag["best_distance"],
                    "best_compat": diag["best_compat"],
                    "elapsed": elapsed,
                }
                self._phase1_log.append(attempt_log)

                if diag["success"]:
                    anchors = _extract_anchors(
                        x, h, self._site_energy_p1, diag, cfg
                    )
                    if anchors is not None:
                        if self.config.verbose:
                            print(
                                f"    ✓ Phase 1 SUCCESS: {anchors.n_anchors} anchor(s) "
                                f"occupying {diag['n_occupied_sites']} site(s), "
                                f"best distance={diag['best_distance']:.2f} Å"
                            )
                        self._anchors = anchors
                        self._phase1_fragment_mol = mol  # ★ store for merge_anchors_to_initial_molecule
                        return anchors

            if self.config.verbose:
                print(
                    f"    ✗ Attempt {attempt + 1} failed "
                    f"(best distance={self._phase1_log[-1]['best_distance']:.2f} Å)"
                )

        if self.config.verbose:
            print("  Phase 1 FAILED: no fragment occupied any HEW site after all attempts.")
        return None

    # ------------------------------------------------------------------
    # Phase 2: Connect
    # ------------------------------------------------------------------

    def phase2_connect(
        self,
        protein_data: dict,
        anchors: AnchorAtoms,
        full_mol_size: int,
        *,
        n_samples: int = 1,
        timesteps: int = 100,
        device: str = "cuda:0",
        anchor_atom_indices: list[int] | None = None,
    ):
        """Run Phase 2: grow a full molecule around the anchor atoms.

        Args:
            protein_data: DrugFlow protein batch dict
            anchors: anchor atoms from Phase 1
            full_mol_size: total number of atoms in the full molecule
            n_samples: number of full molecules to generate
            timesteps: ODE integration steps
            device: torch device
            anchor_atom_indices: which indices (0..full_mol_size-1) are anchors.
                                 If None, anchors are placed at indices 0..n_anchors-1.

        Returns:
            List of RDKit molecules, or None if generation fails.
        """
        cfg = self.config.phase2
        self._site_energy_p2.to(device)
        self._phase2_log = []

        # Default: anchors are the first n_anchors atoms
        if anchor_atom_indices is None:
            anchor_atom_indices = list(range(anchors.n_anchors))

        # ── Build Phase 2 guide function ──
        guide_fn = TwoStageGuideFn(
            site_energy=self._site_energy_p2,
            anchors=anchors,
            config=cfg,
            kts=self._kts_p2,
        ).to(device)
        guide_fn.set_anchor_indices(anchor_atom_indices, full_mol_size)

        # ── Build anchor fixation callback ──
        post_step_callback = None
        pre_step_callback = None
        anchor_fix_mode = getattr(cfg, "anchor_fix_mode", "hard")

        if anchor_fix_mode == "hard":
            # v7.1 default: hard-overwrite every step (per-step reset)
            patch_drugflow_hardfix()
            from guidance.hard_fix import HardFixCallback
            post_step_callback = HardFixCallback(
                anchor_indices=anchor_atom_indices,
                anchor_coords=anchors.positions.clone(),
                anchor_h=anchors.type_probs.clone() if cfg.type_bias_strength > 0 else None,
                fix_coords=True,
                fix_types=(cfg.type_bias_strength > 0),
                verbose=self.config.verbose,
            )

        elif anchor_fix_mode == "hard_locked":
            # ★ Fully locked: anchors removed before ODE, re-inserted after.
            # Zero KPE from anchor atoms — fair baseline for comparison.
            patch_drugflow_hardfix()
            patch_drugflow_fully_locked()  # adds pre_step_callback support
            from guidance.hard_fix import FullyLockedAnchorCallback
            post_step_callback = FullyLockedAnchorCallback(
                anchor_indices=anchor_atom_indices,
                anchor_coords=anchors.positions.clone(),
                anchor_h=anchors.type_probs.clone() if cfg.type_bias_strength > 0 else None,
                verbose=self.config.verbose,
            )
            # Register as both pre_step and post_step callback
            pre_step_callback = post_step_callback

        elif anchor_fix_mode == "annealing":
            # v7.1a: hard-fix → harmonic decay
            patch_drugflow_hardfix()
            from guidance.annealing_fix import AnnealingAnchorFix
            post_step_callback = AnnealingAnchorFix(
                anchor_indices=anchor_atom_indices,
                anchor_coords=anchors.positions.clone(),
                anchor_h=anchors.type_probs.clone() if cfg.type_bias_strength > 0 else None,
                total_steps=timesteps,
                fix_fraction=getattr(cfg, "annealing_fix_fraction", 0.7),
                restraint_start=getattr(cfg, "annealing_restraint_start", 10.0),
                restraint_end=getattr(cfg, "annealing_restraint_end", 0.0),
                ramp=getattr(cfg, "annealing_ramp", "linear"),
                fix_coords=True,
                fix_types=(cfg.type_bias_strength > 0),
                verbose=self.config.verbose,
            )

        elif anchor_fix_mode == "soft":
            # Soft restraint only — guide function handles this via
            # harmonic_restraint_energy in TwoStageGuideFn
            post_step_callback = None

        elif anchor_fix_mode == "kinematic":
            # ★ SV-Flow fusion: kinematic decoupling + CoM-level
            # site-attraction guidance.  Replaces hard coordinate overwrite
            # with smooth, zero-strain translational correction.
            patch_drugflow_hardfix()
            from guidance.kinematic_anchor import KinematicAnchorGuidance
            post_step_callback = KinematicAnchorGuidance(
                anchor_indices=anchor_atom_indices,
                site_energy=self._site_energy_p2,
                total_steps=timesteps,
                lambda_max=getattr(cfg, "kinematic_lambda_max", 0.5),
                profile=getattr(cfg, "kinematic_profile", "quadratic"),
                t_on=getattr(cfg, "kinematic_t_on", 0.5),
                grad_clip=getattr(cfg, "kinematic_grad_clip", 0.5),
                track_kpe=getattr(cfg, "kinematic_track_kpe", True),
                verbose=self.config.verbose,
            )

        else:
            raise ValueError(
                f"Unknown anchor_fix_mode: {anchor_fix_mode!r}. "
                f"Choose 'hard', 'hard_locked', 'annealing', 'soft', or 'kinematic'."
            )

        # ★ Ensure post_step_callback is forwarded by model.sample() → simulate()
        # (monkey-patches in-memory methods; idempotent)
        if post_step_callback is not None:
            from guidance.hard_fix import patch_drugflow_sample_post_step
            patch_drugflow_sample_post_step(self.model)

        if self.config.verbose:
            mode_desc = {
                "hard": "hard-fix every step",
                "hard_locked": "★ fully locked (anchors removed before ODE, zero KPE)",
                "annealing": f"hard-fix ({getattr(cfg, 'annealing_fix_fraction', 0.7):.0%}) "
                             f"→ harmonic decay ({getattr(cfg, 'annealing_restraint_start', 10.0):.1f}"
                             f" → {getattr(cfg, 'annealing_restraint_end', 0.0):.1f})",
                "soft": "harmonic restraint only (k={:.1f})".format(cfg.restraint_force),
                "kinematic": f"kinematic decoupling (λ_max={getattr(cfg, 'kinematic_lambda_max', 0.5)}, "
                             f"profile={getattr(cfg, 'kinematic_profile', 'quadratic')})",
            }
            print(
                f"  Phase 2: growing full molecule ({full_mol_size} atoms, "
                f"{anchors.n_anchors} anchors, λ={cfg.lambda_late})"
            )
            print(f"    anchor_fix_mode: {mode_desc.get(anchor_fix_mode, anchor_fix_mode)}")

        t0 = time.time()
        with torch.no_grad():
            sample_kwargs = dict(
                protein_data=protein_data,
                n_samples=n_samples,
                timesteps=timesteps,
                num_nodes=full_mol_size,
                guide_log_prob=guide_fn if cfg.lambda_late > 0 else None,
                post_step_callback=post_step_callback,
            )
            if pre_step_callback is not None:
                sample_kwargs["pre_step_callback"] = pre_step_callback
            rdmols, trajectories, _ = self.model.sample(**sample_kwargs)
        elapsed = time.time() - t0

        # Log
        valid_count = sum(1 for m in rdmols if m is not None)
        log_entry = {
            "full_mol_size": full_mol_size,
            "n_anchors": anchors.n_anchors,
            "anchor_indices": anchor_atom_indices,
            "n_samples": n_samples,
            "valid_count": valid_count,
            "elapsed": elapsed,
            "anchor_fix_mode": anchor_fix_mode,
        }
        if post_step_callback is not None:
            log_entry["callback_calls"] = post_step_callback.n_calls
            if hasattr(post_step_callback, "n_fix_steps"):
                log_entry["n_fix_steps"] = post_step_callback.n_fix_steps
                log_entry["n_anneal_steps"] = post_step_callback.n_anneal_steps
            # ★ Kinematic mode: capture KPE diagnostics
            if hasattr(post_step_callback, "get_kpe_summary"):
                log_entry["kpe_summary"] = post_step_callback.get_kpe_summary()
            if hasattr(post_step_callback, "get_anchor_trajectory"):
                traj = post_step_callback.get_anchor_trajectory()
                if traj is not None:
                    log_entry["anchor_trajectory_n_frames"] = traj.shape[0]
        self._phase2_log.append(log_entry)

        if self.config.verbose:
            print(f"    Phase 2 done: {valid_count}/{n_samples} valid, {elapsed:.1f}s")

        return rdmols

    # ------------------------------------------------------------------
    # Combined: run both phases
    # ------------------------------------------------------------------

    def generate(
        self,
        protein_data: dict,
        full_mol_size: int,
        *,
        n_phase2_samples: int = 1,
        phase1_timesteps: int = 50,
        phase2_timesteps: int = 100,
        device: str = "cuda:0",
        anchor_atom_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run complete two-stage generation.

        Args:
            protein_data: DrugFlow protein batch dict
            full_mol_size: total atoms in the final molecule
            n_phase2_samples: how many full molecules to generate in Phase 2
            phase1_timesteps: ODE steps for Phase 1
            phase2_timesteps: ODE steps for Phase 2
            device: torch device
            anchor_atom_indices: which atom indices are anchors (default: 0..n_anchors-1)

        Returns:
            dict with keys:
              - "success": bool
              - "anchors": AnchorAtoms or None
              - "molecules": list of RDKit mols or None
              - "phase1_log": list of attempt diagnostics
              - "phase2_log": list of Phase 2 diagnostics
        """
        # Phase 1
        anchors = self.phase1_occupy(
            protein_data,
            ref_mol_size=full_mol_size,
            n_samples_per_attempt=3,
            timesteps=phase1_timesteps,
            device=device,
        )

        generation_mode = "two_stage"  # default

        if anchors is None:
            # ★ Degradation to single-stage KAG (no anchors)
            if self.config.verbose:
                print("  Phase 1 FAILED: degrading to single-stage KAG "
                      "(CoM projection, no anchors).")
            generation_mode = "single_stage_degraded"
            # Create a dummy AnchorAtoms with zero anchors
            anchors = AnchorAtoms(
                positions=torch.zeros(0, 3, device=device),
                type_indices=torch.zeros(0, dtype=torch.long, device=device),
                type_probs=torch.zeros(0, N_ATOM_TYPES, device=device),
                occupied_sites=[],
                compat_scores=[],
                distances=[],
            )
            # Single-stage degradation: skip two-stage pipeline
            # Return immediately — caller should handle via full gradient fallback
            return {
                "success": False,
                "generation_mode": "single_stage_degraded",
                "anchors": None,
                "molecules": None,
                "phase1_log": self._phase1_log,
                "phase2_log": [],
            }
        else:
            # Phase 1 succeeded — prepare initial molecule from fragment
            fragment_mol = self._phase1_fragment_mol
            if fragment_mol is not None and self.config.verbose:
                init_info = merge_anchors_to_initial_molecule(fragment_mol, anchors)
                print(f"  Phase 2 initial state: {init_info['n_fragment_atoms']} "
                      f"fragment atoms, {init_info['n_anchors']} anchors")

        # Phase 2
        molecules = self.phase2_connect(
            protein_data,
            anchors,
            full_mol_size,
            n_samples=n_phase2_samples,
            timesteps=phase2_timesteps,
            device=device,
            anchor_atom_indices=anchor_atom_indices,
        )

        return {
            "success": True,
            "generation_mode": generation_mode,
            "anchors": anchors,
            "molecules": molecules,
            "phase1_log": self._phase1_log,
            "phase2_log": self._phase2_log,
        }

    def get_logs(self) -> dict[str, list[dict]]:
        """Get all generation logs."""
        return {
            "phase1": self._phase1_log,
            "phase2": self._phase2_log,
        }


# ---------------------------------------------------------------------------
# Phase 1 guide function (internal)
# ---------------------------------------------------------------------------


class _Phase1GuideFn:
    """guide_log_prob callable for Phase 1 occupancy.

    Uses strong site-compatibility guidance to pull atoms toward HEW sites,
    with KTS time-shaping for early boost.
    """

    def __init__(
        self,
        site_energy: SiteCompatibilityEnergy,
        lambda_guide: float = 0.5,
        guidance_start: float = 0.05,
        guidance_end: float = 0.95,
        grad_clip: float = 0.5,
        kts: KTSScheduler | None = None,
    ):
        self.site_energy = site_energy
        self.lambda_guide = lambda_guide
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.grad_clip = grad_clip
        self.kts = kts or KTSScheduler()
        self.esfield_lambda = lambda_guide  # for DrugFlow patching compatibility

    def to(self, device: str) -> "_Phase1GuideFn":
        self.site_energy.to(device)
        return self

    def __call__(
        self,
        t_array,
        *,
        x,
        h,
        batch_mask,
        bonds=None,
        bond_types=None,
    ) -> torch.Tensor:
        """Compute Phase 1 guidance energy."""
        n_atoms = x.shape[0]
        device = x.device
        t_val = float(t_array[0] if hasattr(t_array, "__len__") else t_array)

        # Time gating (DrugFlow patching handles this externally, but
        # we double-gate for safety)
        if t_val < self.guidance_start or t_val > self.guidance_end:
            return torch.tensor(0.0, device=device)

        if self.site_energy.n_sites == 0:
            return torch.tensor(0.0, device=device)

        # KTS scaling
        eta = self.kts(t_val)
        lam_eff = self.lambda_guide * eta
        if lam_eff <= 0:
            return torch.zeros(1, device=device)

        # Compute site compatibility energy.
        # Auto-detect whether h is logits or probabilities.
        h_sum = h.sum(dim=-1)
        if (h >= 0).all() and (h <= 1).all() and torch.allclose(h_sum, torch.ones_like(h_sum), atol=0.01):
            atom_probs = h
        else:
            atom_probs = F.softmax(h, dim=-1)
        e_site = self.site_energy(x, atom_type_probs=atom_probs)

        # Return negative energy (DrugFlow convention)
        return -lam_eff * e_site


# ---------------------------------------------------------------------------
# Utility: extract coordinates and type features from RDKit mol
# ---------------------------------------------------------------------------


def _tensors_from_rdmol(
    mol, device: str = "cpu"
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract coordinate and type tensors from an RDKit molecule.

    Maps RDKit atom types to the DrugFlow atom type vocabulary.

    Args:
        mol: RDKit Mol object with 3D conformer
        device: torch device

    Returns:
        (x, h) where x is [n_atoms, 3] coords, h is [n_atoms, n_types]
        one-hot type encodings. Returns (None, None) on failure.
    """
    try:
        from rdkit import Chem

        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None, None

        coords = torch.tensor(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z] for i in range(n_atoms)],
            dtype=torch.float32,
            device=device,
        )

        # One-hot type encoding
        h = torch.zeros(n_atoms, N_ATOM_TYPES, device=device)
        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            is_aromatic = atom.GetIsAromatic()
            type_idx = _rdkit_atomic_to_type_idx(atomic_num, is_aromatic)
            h[i, type_idx] = 1.0

        return coords, h

    except Exception:
        return None, None


def _rdkit_atomic_to_type_idx(atomic_num: int, is_aromatic: bool = False) -> int:
    """Map RDKit atomic number to DrugFlow atom type index."""
    if atomic_num == 6:
        return ATOM_TYPE_TO_IDX["C_aromatic"] if is_aromatic else ATOM_TYPE_TO_IDX["C_sp3"]
    elif atomic_num == 7:
        return ATOM_TYPE_TO_IDX["N_donor"]
    elif atomic_num == 8:
        return ATOM_TYPE_TO_IDX["O_acceptor"]
    elif atomic_num == 16:
        return ATOM_TYPE_TO_IDX["S"]
    elif atomic_num == 15:
        return ATOM_TYPE_TO_IDX["P"]
    elif atomic_num in {9, 17, 35, 53}:
        return ATOM_TYPE_TO_IDX["halogen"]
    elif atomic_num == 5:
        return ATOM_TYPE_TO_IDX["B"]
    else:
        return ATOM_TYPE_TO_IDX["unknown"]
