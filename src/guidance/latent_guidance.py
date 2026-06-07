"""Latent-space site-compatibility energy guidance for v7 two-stage generation.

Provides differentiable energy functions that operate on coordinates x_t (rather
than latent z), since DrugFlow's intermediate coordinates retain geometric
information throughout the denoising trajectory.

Core components:
  - SiteCompatibilityEnergy:  differentiable scalar energy E_site(x_t; sites)
  - apply_latent_guidance:    Metadiffusion-style velocity correction
  - TypeGuidanceBias:         atom-type logit biasing for compatibility

References:
  - Metadiffusion (inference-time meta-energy biasing)
  - Lai et al. (force field guidance on latent)
  - EBMol (zero-shot conditional generation by fixing atoms)
  - Kinetic Path Energy (KPE) + Kinetic Trajectory Shaping (KTS)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dependencies: optional, install with:
#   pip install rdkit openbabel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Atom type vocabulary (consistent with src/models/atom_features.py)
# ---------------------------------------------------------------------------

ATOM_TYPE_VOCAB: tuple[str, ...] = (
    "unknown",
    "C_sp3",
    "C_aromatic",
    "N_donor",
    "N_acceptor",
    "O_acceptor",
    "S",
    "P",
    "halogen",
    "charged",
    "B",
)

ATOM_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ATOM_TYPE_VOCAB)}
N_ATOM_TYPES: int = len(ATOM_TYPE_VOCAB)

# ---------------------------------------------------------------------------
# Site environment types (consistent with src/evaluation/posu.py)
# ---------------------------------------------------------------------------

HEW_ENV_HYDROPHOBIC = "hydrophobic"
HEW_ENV_POLAR_UNSATISFIED = "polar_unsatisfied"
HEW_ENV_MIXED = "mixed"
HEW_ENV_BURIED = "buried"

HEW_ENV_ORDER: list[str] = [
    HEW_ENV_HYDROPHOBIC,
    HEW_ENV_POLAR_UNSATISFIED,
    HEW_ENV_MIXED,
    HEW_ENV_BURIED,
]

HEW_ENV_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(HEW_ENV_ORDER)}

# ---------------------------------------------------------------------------
# Hard-coded compatibility matrix M[env_idx, atom_type_idx]
#
# Design rationale (physical/chemical):
#
#   Hydrophobic sites (mostly apolar contacts, favorable vdW) prefer
#   carbon and halogens.  Polar atoms (O,N) would bring an uncompensated
#   desolvation penalty → negative score.
#
#   Polar-unsatisfied sites (unsatisfied H-bond donors/acceptors on the
#   protein) prefer H-bond-capable ligand atoms (O_acceptor, N_donor,
#   N_acceptor).  Apolar intrusion is mildly penalized.
#
#   Mixed sites accept either type at moderate reward (0.3) — the
#   reduced weight reflects uncertainty.  Buried sites are rarely
#   accessible; only small apolar groups fit, and polar/charged atoms
#   are heavily penalized.
#
#   Score range: +1.0 (strongly compatible) to -1.0 (strongly incompatible).
# ---------------------------------------------------------------------------

_COMPAT_RAW: dict[str, dict[str, float]] = {
    HEW_ENV_HYDROPHOBIC: {
        "C_sp3": 1.0,
        "C_aromatic": 1.0,
        "halogen": 1.0,
        "S": 0.3,
        "P": -0.3,
        "O_acceptor": -0.5,
        "N_donor": -0.5,
        "N_acceptor": -0.5,
        "charged": -1.0,
        "B": -0.3,
        "unknown": -0.5,
    },
    HEW_ENV_POLAR_UNSATISFIED: {
        "O_acceptor": 1.0,
        "N_donor": 1.0,
        "N_acceptor": 1.0,
        "S": 0.3,
        "C_sp3": -0.5,
        "C_aromatic": -0.5,
        "halogen": -0.5,
        "P": -0.3,
        "charged": -1.0,
        "B": -0.3,
        "unknown": -0.5,
    },
    HEW_ENV_MIXED: {
        "C_sp3": 0.5,
        "C_aromatic": 0.5,
        "O_acceptor": 0.5,
        "N_donor": 0.5,
        "N_acceptor": 0.5,
        "S": 0.5,
        "halogen": 0.5,
        "P": 0.1,
        "charged": -0.5,
        "B": 0.1,
        "unknown": 0.0,
    },
    HEW_ENV_BURIED: {
        "C_sp3": 0.3,
        "halogen": 0.5,
        "C_aromatic": -0.5,
        "S": -0.3,
        "O_acceptor": -1.0,
        "N_donor": -1.0,
        "N_acceptor": -1.0,
        "P": -0.5,
        "charged": -1.0,
        "B": -0.3,
        "unknown": -0.5,
    },
}


def _build_compat_tensor() -> torch.Tensor:
    """Build [n_envs, n_atom_types] compatibility matrix from raw dict."""
    mat = torch.zeros(len(HEW_ENV_ORDER), N_ATOM_TYPES)
    for env_idx, env_name in enumerate(HEW_ENV_ORDER):
        env_scores = _COMPAT_RAW.get(env_name, {})
        for atom_type, score in env_scores.items():
            if atom_type in ATOM_TYPE_TO_IDX:
                mat[env_idx, ATOM_TYPE_TO_IDX[atom_type]] = score
    return mat


# Singleton — built once at import time.
COMPAT_MATRIX: torch.Tensor = _build_compat_tensor()


# ---------------------------------------------------------------------------
# Environment classification (same logic as src/evaluation/posu.py and
# src/models/analytic_esfield.py)
# ---------------------------------------------------------------------------

def classify_hew_environment(site: dict) -> str:
    """Classify a HEW site by its local protein environment.

    Uses site metadata fields populated during crystal-water analysis:
      - hbond_count: number of H-bonds the water makes
      - hydrophobic_contact_count: number of apolar contacts
      - nearest_protein_distance: closest protein heavy-atom distance

    Rules (in priority order):
      1. nearest_protein_distance < 2.5 Å → buried
      2. hydrophobic_contact_count >= 4 AND hbond_count <= 1 → hydrophobic
      3. hbond_count <= 1 AND hydrophobic_contact_count <= 2 → polar_unsatisfied
      4. Everything else → mixed
    """
    features = site.get("features", {})
    hbond = features.get("hbond_count", 0)
    hydrophobic = features.get("hydrophobic_contact_count", 0)
    nearest_dist = features.get("nearest_protein_distance", 4.0)

    if nearest_dist < 2.5:
        return HEW_ENV_BURIED
    if hydrophobic >= 4 and hbond <= 1:
        return HEW_ENV_HYDROPHOBIC
    if hbond <= 1 and hydrophobic <= 2:
        return HEW_ENV_POLAR_UNSATISFIED
    return HEW_ENV_MIXED


def hew_env_to_idx(env: str) -> int:
    """Map environment string to integer index for tensor lookup."""
    return HEW_ENV_TO_IDX.get(env, 2)  # default: mixed


# ---------------------------------------------------------------------------
# SiteCompatibilityEnergy
# ---------------------------------------------------------------------------

@dataclass
class SiteCompatibilityEnergy:
    """Differentiable site-compatibility energy function.

    Computes a scalar energy E(x_t; sites) whose gradient w.r.t. atom
    coordinates x_t pulls compatible atoms toward candidate HEW sites
    and pushes incompatible atoms away.

    Energy definition:
      E = - Σ_site Σ_atom [
            compat(atom_type, site_type) * exp(-d_ij² / (2σ²))
          ]

    where:
      - d_ij is the Euclidean distance between atom i and site center j
      - compat ∈ [-1, +1] comes from the hard-coded compatibility matrix
      - σ is the Gaussian kernel width (default 3.0 Å)

    Physical intuition:
      A compatible atom near a site contributes a negative term (lowering
      the energy → favorable).  An incompatible atom near a site contributes
      a positive term (raising the energy → unfavorable, pushing it away).

      The Gaussian kernel creates a smooth potential well centered on each
      site, whose width σ controls the capture radius.
    """

    sigma_distance: float = 3.0
    """Gaussian kernel width in Angstrom.  Larger values create a wider
    attractive funnel; smaller values require precise placement."""

    compatibility_matrix: torch.Tensor | None = None
    """[n_envs, n_atom_types] compatibility scores.  If None, uses the
    global COMPAT_MATRIX singleton."""

    # Internal state (set by to())
    _device: str = "cpu"
    _site_centers: torch.Tensor | None = None   # [n_sites, 3]
    _site_env_indices: torch.Tensor | None = None  # [n_sites]
    _site_confs: torch.Tensor | None = None     # [n_sites]

    def __post_init__(self):
        if self.compatibility_matrix is None:
            self.compatibility_matrix = COMPAT_MATRIX.clone()

    # -- Site registration ------------------------------------------------

    def register_sites(
        self,
        centers: torch.Tensor,
        env_indices: torch.Tensor,
        confidences: torch.Tensor | None = None,
    ) -> None:
        """Register candidate HEW sites for energy computation.

        Args:
            centers: [n_sites, 3] site center coordinates
            env_indices: [n_sites] HEW environment indices
            confidences: [n_sites] optional confidence weights
        """
        self._site_centers = centers.to(device=self._device, dtype=torch.float32)
        self._site_env_indices = env_indices.to(device=self._device, dtype=torch.long)
        if confidences is not None:
            self._site_confs = confidences.to(device=self._device, dtype=torch.float32)
        else:
            self._site_confs = torch.ones(len(centers), device=self._device)

    @property
    def n_sites(self) -> int:
        if self._site_centers is None:
            return 0
        return self._site_centers.shape[0]

    def to(self, device: str) -> "SiteCompatibilityEnergy":
        """Move internal tensors to device."""
        self._device = device
        self.compatibility_matrix = self.compatibility_matrix.to(device)
        if self._site_centers is not None:
            self._site_centers = self._site_centers.to(device)
            self._site_env_indices = self._site_env_indices.to(device)
            if self._site_confs is not None:
                self._site_confs = self._site_confs.to(device)
        return self

    # -- Energy computation -----------------------------------------------

    def __call__(
        self,
        x_t: torch.Tensor,
        atom_type_probs: torch.Tensor | None = None,
        atom_type_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scalar site-compatibility energy E(x_t).

        Args:
            x_t: [n_atoms, 3] current atom coordinates
            atom_type_probs: [n_atoms, n_atom_types] soft type probabilities
                             (e.g., from softmax over model logits). If None,
                             atom_type_indices must be provided.
            atom_type_indices: [n_atoms] hard type assignments (long tensor).
                               Only used when atom_type_probs is None.

        Returns:
            Scalar energy (lower = more compatible).  Differentiable w.r.t. x_t.
        """
        if self._site_centers is None or self.n_sites == 0:
            return x_t.new_zeros(())

        n_atoms = x_t.shape[0]
        n_sites = self.n_sites

        # -- 1. Pairwise distances [n_atoms, n_sites] ---------------------
        rel = x_t[:, None, :] - self._site_centers[None, :, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)

        # -- 2. Gaussian kernel weight: exp(-d² / (2σ²)) ------------------
        sigma2 = 2.0 * self.sigma_distance ** 2
        gauss_weight = torch.exp(-(dist ** 2) / sigma2)  # [n_atoms, n_sites]

        # -- 3. Compatibility scores per atom-site pair -------------------
        if atom_type_probs is not None:
            # Soft: expected compatibility under type distribution
            n_types = min(atom_type_probs.shape[-1], N_ATOM_TYPES)
            probs = atom_type_probs[:, :n_types]  # [n_atoms, n_types]
            env_compat = self.compatibility_matrix[
                self._site_env_indices, :
            ][:, :n_types]  # [n_sites, n_types]
            compat = torch.matmul(probs, env_compat.T)  # [n_atoms, n_sites]
        elif atom_type_indices is not None:
            # Hard: lookup compatibility for each atom's assigned type
            compat = self.compatibility_matrix[
                self._site_env_indices, :
            ][:, atom_type_indices].T  # [n_atoms, n_sites]
        else:
            raise ValueError(
                "Either atom_type_probs or atom_type_indices must be provided"
            )

        # -- 4. Confidence weighting --------------------------------------
        if self._site_confs is not None:
            gauss_weight = gauss_weight * self._site_confs[None, :]

        # -- 5. Energy = -Σ_i Σ_j compat_ij * gauss(d_ij) -----------------
        #
        # Compatible atom near site:  compat > 0, gauss > 0 → energy < 0 (good)
        # Incompatible atom near site: compat < 0, gauss > 0 → energy > 0 (bad)
        #
        pair_contributions = compat * gauss_weight  # [n_atoms, n_sites]
        energy = -pair_contributions.sum()

        return energy

    def compute_per_site_scores(
        self,
        x_t: torch.Tensor,
        atom_type_probs: torch.Tensor | None = None,
        atom_type_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-site compatibility scores (for diagnostics).

        Returns:
            [n_sites] tensor with score per site.  Higher = better occupied.
        """
        if self._site_centers is None or self.n_sites == 0:
            return x_t.new_zeros(0)

        n_atoms = x_t.shape[0]

        rel = x_t[:, None, :] - self._site_centers[None, :, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)

        sigma2 = 2.0 * self.sigma_distance ** 2
        gauss_weight = torch.exp(-(dist ** 2) / sigma2)

        if atom_type_probs is not None:
            n_types = min(atom_type_probs.shape[-1], N_ATOM_TYPES)
            probs = atom_type_probs[:, :n_types]
            env_compat = self.compatibility_matrix[
                self._site_env_indices, :
            ][:, :n_types]
            compat = torch.matmul(probs, env_compat.T)
        elif atom_type_indices is not None:
            compat = self.compatibility_matrix[
                self._site_env_indices, :
            ][:, atom_type_indices].T
        else:
            raise ValueError("atom_type_probs or atom_type_indices required")

        # Per-site sum over atoms of compat * gauss
        per_site = (compat * gauss_weight).sum(dim=0)  # [n_sites]
        if self._site_confs is not None:
            per_site = per_site * self._site_confs

        return per_site  # higher = more compatible occupancy

    def compute_gradient(
        self,
        x_t: torch.Tensor,
        atom_type_probs: torch.Tensor | None = None,
        atom_type_indices: torch.Tensor | None = None,
        grad_clip: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and its gradient w.r.t. coordinates.

        Args:
            x_t: [n_atoms, 3] coordinates (detached copy will be made)
            grad_clip: max per-atom gradient norm

        Returns:
            (energy_value, gradient) where gradient has shape [n_atoms, 3]
        """
        coords = x_t.detach().clone().requires_grad_(True)
        energy = self(coords, atom_type_probs=atom_type_probs,
                      atom_type_indices=atom_type_indices)
        grad = torch.autograd.grad(
            energy, coords, create_graph=False, retain_graph=False
        )[0]
        # Per-atom gradient clipping
        if grad_clip > 0:
            gnorm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.clamp(grad_clip / gnorm, max=1.0)
            grad = grad * scale
        return energy.detach(), grad.detach()


# ---------------------------------------------------------------------------
# apply_latent_guidance — Metadiffusion-style velocity correction
# ---------------------------------------------------------------------------

def apply_latent_guidance(
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    site_energy: SiteCompatibilityEnergy,
    t: float,
    *,
    atom_type_probs: torch.Tensor | None = None,
    atom_type_indices: torch.Tensor | None = None,
    lambda_guide: float = 0.5,
    grad_clip: float = 1.0,
    kts_eta: float = 1.0,
    dt: float | None = None,
) -> dict[str, Any]:
    """Apply latent site-compatibility guidance to a flow-matching velocity.

    Follows the Metadiffusion prescription for inference-time energy biasing:

        x_{t+Δt} = x_t + Δt * η(t) * (v_pred - λ * ∇_x E_site)

    where η(t) is the KTS time-scaling factor (default 1.0 = no scaling).

    IMPORTANT: The gradient is applied ONLY to coordinates because atom
    types cannot be directly differentiated in the current framework.
    Type biasing is handled separately via TypeGuidanceBias (see below).

    Args:
        x_t:      [n_atoms, 3] current atom coordinates at step t
        v_pred:   [n_atoms, 3] predicted velocity from the flow model
        site_energy: SiteCompatibilityEnergy instance with registered sites
        t:        current time (0 = noise, 1 = data)
        atom_type_probs: [n_atoms, n_types] soft type probabilities
        atom_type_indices: [n_atoms] hard type assignments
        lambda_guide: guidance strength (0 = no guidance)
        grad_clip: max gradient norm per atom
        kts_eta:   KTS time-scaling factor η(t)
        dt:        integration step size (if None, assume caller handles
                   the Euler step separately)

    Returns:
        dict with keys:
          - "guided_velocity": [n_atoms, 3] corrected velocity
          - "energy": float, site-compatibility energy value
          - "grad_norm": float, max per-atom gradient norm
          - "lambda_effective": float, λ * η(t)
    """
    if lambda_guide <= 0 or site_energy.n_sites == 0:
        return {
            "guided_velocity": v_pred,
            "energy": 0.0,
            "grad_norm": 0.0,
            "lambda_effective": 0.0,
        }

    energy, grad = site_energy.compute_gradient(
        x_t,
        atom_type_probs=atom_type_probs,
        atom_type_indices=atom_type_indices,
        grad_clip=grad_clip,
    )

    # Effective guidance strength with KTS scaling
    lam_eff = lambda_guide * kts_eta

    # Metadiffusion velocity correction:
    #   v_guided = v_pred - λ * ∇_x E_site
    # So that dx/dt = v_guided moves atoms toward lower energy.
    guided_v = v_pred - lam_eff * grad

    return {
        "guided_velocity": guided_v,
        "energy": float(energy.cpu()),
        "grad_norm": float(grad.norm(dim=-1).max().cpu()),
        "lambda_effective": lam_eff,
    }


# ---------------------------------------------------------------------------
# TypeGuidanceBias — atom-type logit biasing
# ---------------------------------------------------------------------------

class TypeGuidanceBias:
    """Add compatibility-based bias to atom-type logits during sampling.

    Since the gradient from SiteCompatibilityEnergy only affects coordinates,
    this class provides a complementary mechanism to nudge atom types toward
    compatibility with nearby candidate HEW sites.

    At each denoising step, for each atom i:
      logits'_i = logits_i + β * Σ_j compat_bias(type, site_j) * gauss(d_ij)

    where:
      - β = λ_type controls the bias strength
      - compat_bias is the raw compatibility row for site_j's environment
      - gauss(d_ij) down-weights distant sites

    This is a SOFT bias: it increases the probability of compatible types
    without forcing a hard assignment.  The temperature of the softmax
    implicitly controls how strongly the bias affects sampling.
    """

    def __init__(
        self,
        site_energy: SiteCompatibilityEnergy,
        lambda_type: float = 0.1,
        sigma_distance: float | None = None,
    ):
        """Initialize type guidance.

        Args:
            site_energy: SiteCompatibilityEnergy with registered sites
            lambda_type: bias strength (0 = no bias)
            sigma_distance: Gaussian width for distance weighting
                            (default: same as site_energy.sigma_distance)
        """
        self._site_energy = site_energy
        self.lambda_type = lambda_type
        self.sigma_distance = (
            sigma_distance
            if sigma_distance is not None
            else site_energy.sigma_distance
        )

    def __call__(
        self,
        logits: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Apply type bias to logits.

        Args:
            logits: [n_atoms, n_atom_types] raw type logits from model
            x_t:    [n_atoms, 3] current atom coordinates

        Returns:
            [n_atoms, n_atom_types] biased logits (same shape)
        """
        if self.lambda_type <= 0 or self._site_energy.n_sites == 0:
            return logits

        n_atoms = x_t.shape[0]
        n_types = logits.shape[-1]
        n_sites = self._site_energy.n_sites

        # Get site centers and env indices
        centers = self._site_energy._site_centers  # [n_sites, 3]
        env_indices = self._site_energy._site_env_indices  # [n_sites]

        # Pairwise distances
        rel = x_t[:, None, :] - centers[None, :, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)  # [n_atoms, n_sites]

        # Gaussian distance weight
        sigma2 = 2.0 * self.sigma_distance ** 2
        gauss_w = torch.exp(-(dist ** 2) / sigma2)  # [n_atoms, n_sites]

        # For each site j, get compatibility vector [n_atom_types]
        # compat_matrix is [n_envs, n_atom_types]
        compat = self._site_energy.compatibility_matrix.to(
            device=logits.device, dtype=logits.dtype
        )
        site_compat = compat[env_indices, :]  # [n_sites, n_atom_types]

        # Per-atom bias = Σ_j gauss(d_ij) * compat[env_j, :]
        # Shape: [n_atoms, n_sites] @ [n_sites, n_types] → [n_atoms, n_types]
        bias = torch.matmul(gauss_w, site_compat[:, :n_types])

        # Scale by λ_type
        return logits + self.lambda_type * bias


# ---------------------------------------------------------------------------
# Convenience: build SiteCompatibilityEnergy from site_map dict
# ---------------------------------------------------------------------------

def build_site_energy_from_map(
    site_map: dict[str, Any],
    *,
    sigma_distance: float = 3.0,
    min_confidence: float = 0.0,
    enabled_envs: tuple[str, ...] = (
        HEW_ENV_HYDROPHOBIC,
        HEW_ENV_POLAR_UNSATISFIED,
        HEW_ENV_MIXED,
    ),
    top_k: int = 0,
) -> SiteCompatibilityEnergy:
    """Build a SiteCompatibilityEnergy from a raw site_map dict.

    Filters sites by environment type, confidence, and optionally takes
    the top-k by confidence.

    Args:
        site_map: dict with key "sites" containing site objects
        sigma_distance: Gaussian kernel width
        min_confidence: minimum site confidence to include
        enabled_envs: which HEW environments to include
        top_k: if > 0, keep only the top-k highest-confidence HEW sites

    Returns:
        SiteCompatibilityEnergy with sites registered and ready to use.
    """
    energy = SiteCompatibilityEnergy(sigma_distance=sigma_distance)

    sites = site_map.get("sites", [])
    hew_sites = [s for s in sites if s.get("site_type") == "high_energy_water"]

    if not hew_sites:
        return energy  # no HEW sites → energy always 0

    # Filter and classify
    candidates = []
    for s in hew_sites:
        env = classify_hew_environment(s)
        conf = s.get("confidence", 1.0)
        if env not in enabled_envs:
            continue
        if conf < min_confidence:
            continue
        candidates.append((s, env, conf))

    if not candidates:
        return energy

    # Optional top-k by confidence
    if top_k > 0 and len(candidates) > top_k:
        candidates.sort(key=lambda x: -x[2])
        candidates = candidates[:top_k]

    # Build tensors
    centers = torch.tensor([c[0]["center"] for c in candidates], dtype=torch.float32)
    env_indices = torch.tensor(
        [hew_env_to_idx(c[1]) for c in candidates], dtype=torch.long
    )
    confidences = torch.tensor([c[2] for c in candidates], dtype=torch.float32)

    energy.register_sites(centers, env_indices, confidences)
    return energy


# ---------------------------------------------------------------------------
# Harmornic restraint helper (for Phase 2 fixed atoms)
# ---------------------------------------------------------------------------

def harmonic_restraint_energy(
    x_t: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_positions: torch.Tensor,
    force_constant: float = 10.0,
) -> torch.Tensor:
    """Compute harmonic restraint energy for fixed atoms.

    E_restraint = k * Σ_{i ∈ fixed} ||x_i - x_i^fixed||²

    Args:
        x_t: [n_atoms, 3] current coordinates
        fixed_mask: [n_atoms] bool, True for fixed atoms
        fixed_positions: [n_atoms, 3] target positions (only used where mask=True)
        force_constant: harmonic force constant k (kcal/mol/Å² equivalent)

    Returns:
        Scalar restraint energy.  Differentiable w.r.t. x_t.
    """
    if not fixed_mask.any():
        return x_t.new_zeros(())
    delta = x_t[fixed_mask] - fixed_positions[fixed_mask]
    return force_constant * (delta ** 2).sum()
