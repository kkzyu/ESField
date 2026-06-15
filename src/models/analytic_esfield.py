"""Analytic ESField v6-D.2 — Capture-to-Displacement Reward Guidance.

v6-D.2 replaces the single narrow Gaussian with a two-stage reward:
  E_reward = -sum_j B_j * SoftMaxAgg_i(q_ij(t))

  q_ij(t) = M_ij * [lambda_cap(t) * C(d_ij) + lambda_occ(t) * O(d_ij)]

  C(d) = exp(-d^2 / (2 * sigma_cap^2)),  sigma_cap = 2.5 Å   (capture funnel)
  O(d) = exp(-d^2 / (2 * sigma_occ^2)),  sigma_occ = 1.0 Å   (direct occupancy)

Both Gaussians are centered at d=0 (water center), not d0=3.0.
The capture term pulls atoms from 3-5 Å toward the HEW center.
The occupancy term rewards atoms that reach d < 2 Å.

E_guide = E_reward + w_wrong * E_wrong + w_clash * E_clash + w_overfill * E_overfill

First round: only high-confidence hydrophobic candidate HEW enabled.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Atom type vocabulary (consistent with atom_features.py)
# ---------------------------------------------------------------------------

ATOM_TYPE_TO_IDX: dict[str, int] = {
    "unknown": 0, "C_sp3": 1, "C_aromatic": 2,
    "N_donor": 3, "N_acceptor": 4, "O_acceptor": 5,
    "S": 6, "P": 7, "halogen": 8, "charged": 9, "B": 10,
}
N_ATOM_TYPES = len(ATOM_TYPE_TO_IDX)

# ---------------------------------------------------------------------------
# HEW environment types
# ---------------------------------------------------------------------------

HEW_ENV_HYDROPHOBIC = "hydrophobic"
HEW_ENV_POLAR_UNSATISFIED = "polar_unsatisfied"
HEW_ENV_MIXED = "mixed"
HEW_ENV_BURIED = "buried"

HEW_ENV_ORDER = [HEW_ENV_HYDROPHOBIC, HEW_ENV_POLAR_UNSATISFIED, HEW_ENV_MIXED, HEW_ENV_BURIED]

# ---------------------------------------------------------------------------
# Heuristic compatibility matrix M(atom_type, hew_environment)
# ---------------------------------------------------------------------------

# ── DEPRECATED: Use guidance.latent_guidance.COMPAT_MATRIX instead ──
# This matrix is kept for backward compatibility only.
# The authoritative matrix matching paper Appendix Table 10 is in latent_guidance.py.
_HEW_COMPATIBILITY_RAW: dict[str, dict[str, float]] = {
    HEW_ENV_HYDROPHOBIC: {
        "C_sp3": 1.0, "C_aromatic": 1.0,
        "S": 0.3, "halogen": 0.3,
        "O_acceptor": -0.5, "N_donor": -0.5, "N_acceptor": -0.5,
        "charged": -1.0, "P": -0.3,
    },
    HEW_ENV_POLAR_UNSATISFIED: {
        "O_acceptor": 1.0, "N_donor": 1.0, "N_acceptor": 1.0,
        "S": 0.3,
        "C_sp3": -0.3, "C_aromatic": -0.3, "halogen": -0.3,
        "charged": -1.0, "P": -0.3,
    },
    HEW_ENV_MIXED: {
        "C_sp3": 0.3, "C_aromatic": 0.3,
        "O_acceptor": 0.3, "N_donor": 0.3, "N_acceptor": 0.3,
        "S": 0.2, "halogen": 0.2,
        "charged": -0.5, "P": 0.1,
    },
    HEW_ENV_BURIED: {
        "C_sp3": 0.3, "halogen": 0.5,
        "S": -0.3,
        "O_acceptor": -1.0, "N_donor": -1.0, "N_acceptor": -1.0,
        "C_aromatic": -0.5, "charged": -1.0, "P": -0.5,
    },
}


def _build_compat_tensor() -> torch.Tensor:
    mat = torch.zeros(len(HEW_ENV_ORDER), N_ATOM_TYPES)
    for env_idx, env_name in enumerate(HEW_ENV_ORDER):
        env_scores = _HEW_COMPATIBILITY_RAW.get(env_name, {})
        for atom_type, score in env_scores.items():
            if atom_type in ATOM_TYPE_TO_IDX:
                mat[env_idx, ATOM_TYPE_TO_IDX[atom_type]] = score
    return mat


COMPAT_MATRIX = _build_compat_tensor()


def classify_hew_environment(site: dict) -> str:
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
    try:
        return HEW_ENV_ORDER.index(env)
    except ValueError:
        return 2


HEW_ENV_BONUS = {
    HEW_ENV_HYDROPHOBIC: 1.0,
    HEW_ENV_POLAR_UNSATISFIED: 1.0,
    HEW_ENV_MIXED: 0.4,
    HEW_ENV_BURIED: 0.3,
}

# ---------------------------------------------------------------------------
# v6-D.2 Configuration
# ---------------------------------------------------------------------------

# Time schedule: DrugFlow t goes from 0 (noise) to 1 (data).
# Guidance is active in [guidance_start, guidance_end].
# Within that window:
#   middle: capture strong, occupancy weak
#   late:   capture weak, occupancy strong
#   final:  reduce overall force

@dataclass
class TimeSchedule:
    """Piecewise-linear schedule for capture and occupancy weights over t in [0,1]."""
    # Phase boundaries (fraction of guidance window)
    cap_rise_end: float = 0.2      # capture ramps up from 0 to cap_peak
    cap_peak_end: float = 0.55     # capture stays at peak
    cap_fall_start: float = 0.55   # capture starts to fall
    cap_fall_end: float = 0.85     # capture near zero

    occ_rise_start: float = 0.35   # occupancy starts to rise
    occ_peak_start: float = 0.65   # occupancy reaches peak
    occ_fall_start: float = 0.85   # occupancy starts to fall
    occ_fall_end: float = 1.0      # occupancy near zero

    cap_peak: float = 1.0
    occ_peak: float = 1.0

    # Overall force reduction near end
    global_fall_start: float = 0.85
    global_fall_end: float = 1.0

    def get_weights(self, t_fraction: float) -> tuple[float, float, float]:
        """Return (lambda_cap, lambda_occ, global_scale) for a given t_fraction in [0,1]."""
        tf = max(0.0, min(1.0, t_fraction))

        def _ramp(x, x0, y0, x1, y1):
            if x <= x0:
                return y0
            if x >= x1:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        cap = _ramp(tf, 0.0, 0.0, self.cap_rise_end, self.cap_peak)
        cap = _ramp(tf, self.cap_fall_start, cap, self.cap_fall_end, 0.0)

        occ = _ramp(tf, self.occ_rise_start, 0.0, self.occ_peak_start, self.occ_peak)
        occ = _ramp(tf, self.occ_fall_start, occ, self.occ_fall_end, 0.0)

        global_scale = _ramp(tf, self.global_fall_start, 1.0, self.global_fall_end, 0.3)

        return cap, occ, global_scale


@dataclass
class V6D2Config:
    """Configuration for Analytic ESField v6-D.2 guidance."""

    # --- Capture and occupancy Gaussians (both centered at d=0) ---
    sigma_cap: float = 2.5          # Capture funnel width (Angstrom)
    sigma_occ: float = 1.0          # Direct occupancy reward width (Angstrom)
    direct_occ_threshold: float = 2.0  # d < this = direct displacement (for logging)

    # --- SoftMax aggregation ---
    softmax_tau: float = 1.0        # Temperature for SoftMaxAgg over atoms per site

    # --- Protection weights ---
    wrong_atom_weight: float = 0.5
    clash_weight: float = 0.0       # Disabled without protein coords
    overfill_weight: float = 0.3
    bond_strain_weight: float = 0.0  # Reserved

    # --- Clash parameters ---
    clash_distance: float = 2.0
    clash_sigma: float = 0.3

    # --- Overfill ---
    overfill_max_per_site: int = 2

    # --- candidate HEW filtering ---
    min_confidence: float = 0.7
    enabled_envs: tuple[str, ...] = (HEW_ENV_HYDROPHOBIC,)
    top_k: int = 5                  # Max candidate HEW per pocket (0 = all enabled)

    # --- Guidance schedule ---
    guidance_start: float = 0.3     # Start earlier (capture needs time)
    guidance_end: float = 0.88
    esfield_lambda: float = 0.5
    grad_clip: float = 1.0

    # --- Time schedule ---
    time_schedule: TimeSchedule = field(default_factory=TimeSchedule)

    # --- Smooth cutoff ---
    cutoff_dist: float = 6.0
    cutoff_smooth_width: float = 0.5

    # --- Logging ---
    log_gradient_norms: bool = False


# ---------------------------------------------------------------------------
# AnalyticESFieldGuideV2 — v6-D.2
# ---------------------------------------------------------------------------

class AnalyticESFieldGuideV2:
    """v6-D.2: Capture-to-Displacement Analytic Guidance.

    Drop-in replacement for ESFieldGuide / AnalyticESFieldGuide.
    """

    SITE_TYPE_MAP = {"unknown": 0, "high_energy_water": 1, "stable_water": 2, "hydrophobic_cavity": 3}

    def __init__(
        self,
        site_map: dict | str | Path,
        config: V6D2Config | None = None,
        protein_coords: torch.Tensor | None = None,
    ):
        self.config = config or V6D2Config()

        self.guidance_start = self.config.guidance_start
        self.guidance_end = self.config.guidance_end
        self.esfield_lambda = self.config.esfield_lambda
        self.grad_clip = self.config.grad_clip

        _sm = site_map if isinstance(site_map, dict) else json.loads(Path(site_map).read_text())
        self._all_sites = _sm["sites"]
        self._protein_coords = protein_coords

        self._hew_indices, self._hew_sites, self._hew_envs = self._filter_candidate_hews()
        self._n_hew = len(self._hew_sites)

        # Per-site tensors (filled in to())
        self._hew_centers = None     # [n_hew, 3]
        self._hew_radii = None       # [n_hew]
        self._hew_confs = None       # [n_hew]
        self._hew_env_indices = None # [n_hew]
        self._hew_bonus = None       # [n_hew]

        # Logging accumulator
        self._log: dict[str, list[float]] = {}

    def _filter_candidate_hews(self):
        """Filter to candidate HEW: enabled envs, high confidence, non-buried."""
        cfg = self.config
        candidates = []
        sites_out = []
        envs_out = []

        for idx, site in enumerate(self._all_sites):
            if site.get("site_type") != "high_energy_water":
                continue
            env = classify_hew_environment(site)
            conf = site.get("confidence", 1.0)

            # Only enabled environments
            if env not in cfg.enabled_envs:
                continue

            # High confidence threshold
            if conf < cfg.min_confidence:
                continue

            candidates.append((idx, site, env))

        # Top-k by confidence × bonus
        if cfg.top_k > 0 and len(candidates) > cfg.top_k:
            scored = [(s.get("confidence", 1.0) * HEW_ENV_BONUS.get(e, 0.3), idx, s, e)
                      for idx, s, e in candidates]
            scored.sort(key=lambda x: -x[0])
            candidates = [(idx, s, e) for _, idx, s, e in scored[: cfg.top_k]]

        for idx, s, e in candidates:
            sites_out.append(s)
            envs_out.append(e)

        return [c[0] for c in candidates], sites_out, envs_out

    @property
    def n_hew(self) -> int:
        return self._n_hew

    def to(self, device):
        n = self._n_hew
        if n == 0:
            self._hew_centers = torch.zeros(0, 3, device=device)
            self._hew_radii = torch.zeros(0, device=device)
            self._hew_confs = torch.zeros(0, device=device)
            self._hew_env_indices = torch.zeros(0, dtype=torch.long, device=device)
            self._hew_bonus = torch.zeros(0, device=device)
        else:
            self._hew_centers = torch.tensor(
                [s["center"] for s in self._hew_sites], dtype=torch.float32, device=device)
            self._hew_radii = torch.tensor(
                [s.get("radius", 1.4) for s in self._hew_sites], dtype=torch.float32, device=device)
            self._hew_confs = torch.tensor(
                [s.get("confidence", 1.0) for s in self._hew_sites], dtype=torch.float32, device=device)
            self._hew_env_indices = torch.tensor(
                [hew_env_to_idx(e) for e in self._hew_envs], dtype=torch.long, device=device)
            self._hew_bonus = torch.tensor(
                [HEW_ENV_BONUS.get(e, 0.3) for e in self._hew_envs], dtype=torch.float32, device=device)

        if self._protein_coords is not None:
            self._protein_coords = self._protein_coords.to(device)

        self._compat = COMPAT_MATRIX.to(device=device, dtype=torch.float32)
        self._log = {}
        return self

    # ------------------------------------------------------------------
    # Core energy computation
    # ------------------------------------------------------------------

    def _compute_both_gaussians(self, dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute capture C(d) and occupancy O(d), both centered at d=0."""
        sigma_cap = self.config.sigma_cap
        sigma_occ = self.config.sigma_occ
        C = torch.exp(-(dist ** 2) / (2.0 * sigma_cap ** 2))
        O = torch.exp(-(dist ** 2) / (2.0 * sigma_occ ** 2))
        return C, O

    def _smooth_cutoff(self, dist: torch.Tensor) -> torch.Tensor:
        cutoff = self.config.cutoff_dist
        width = max(self.config.cutoff_smooth_width, 0.05)
        return torch.sigmoid((cutoff - dist) / width)

    def _compute_compatibility(self, atom_probs: torch.Tensor, hew_env_indices: torch.Tensor) -> torch.Tensor:
        n_atom_types = min(atom_probs.shape[-1], N_ATOM_TYPES)
        probs = atom_probs[:, :n_atom_types]
        env_compat = self._compat[hew_env_indices, :n_atom_types]
        return torch.matmul(probs, env_compat.T)  # [n_atoms, n_hew]

    def _compute_time_weights(self, t_scalar: float) -> tuple[float, float, float]:
        """Get (lambda_cap, lambda_occ, global_scale) for current time step."""
        t_val = float(t_scalar)
        # Map from [guidance_start, guidance_end] to [0, 1]
        gs = self.config.guidance_start
        ge = self.config.guidance_end
        if t_val <= gs:
            return 0.0, 0.0, 0.0
        if t_val >= ge:
            t_frac = 1.0
        else:
            t_frac = (t_val - gs) / (ge - gs)
        return self.config.time_schedule.get_weights(t_frac)

    def _softmax_agg(self, q_ij: torch.Tensor) -> torch.Tensor:
        """SoftMaxAgg over atoms for each site: tau * log(sum_i exp(q_ij / tau)).

        Args:
            q_ij: [n_atoms, n_hew] quality scores

        Returns:
            agg_q: [n_hew] aggregated quality per site
        """
        tau = max(self.config.softmax_tau, 0.01)
        # log-sum-exp over atoms (dim=0)
        agg_q = tau * torch.logsumexp(q_ij / tau, dim=0)  # [n_hew]
        return agg_q

    # ------------------------------------------------------------------
    # Main callable interface
    # ------------------------------------------------------------------

    def __call__(self, t_array, *, x, h, batch_mask, bonds=None, bond_types=None) -> torch.Tensor:
        n_atoms = x.shape[0]
        n_hew = self._n_hew
        if n_hew == 0:
            return (x * 0).sum()

        # Time weights
        t_val = float(t_array[0] if hasattr(t_array, '__len__') else t_array)
        lam_cap, lam_occ, global_scale = self._compute_time_weights(t_val)
        if global_scale == 0.0:
            return (x * 0).sum()

        # 1. Pairwise distances
        rel = x[:, None, :] - self._hew_centers[None, :, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)

        # 2. Capture and occupancy Gaussians (both d=0 centered)
        C, O = self._compute_both_gaussians(dist)

        # Combined occupancy for protection terms
        occ_combined = lam_cap * C + lam_occ * O

        # 3. Smooth cutoff
        cutoff = self._smooth_cutoff(dist)

        # 4. Atom type probabilities and compatibility
        atom_probs = F.softmax(h, dim=-1)
        compat = self._compute_compatibility(atom_probs, self._hew_env_indices)

        # 5. Reward: M_reward * [lambda_cap * C + lambda_occ * O]
        m_reward = F.softplus(compat) / F.softplus(torch.tensor(1.0, device=compat.device))
        q_ij = m_reward * (lam_cap * C + lam_occ * O) * cutoff * self._hew_confs[None, :]

        # 6. SoftMaxAgg over atoms → per-site reward
        agg_q = self._softmax_agg(q_ij)  # [n_hew]
        e_reward = -(self._hew_bonus * agg_q).sum()

        # 7. Protection terms
        # E_wrong: penalty for incompatible atoms
        m_penalty = F.softplus(-compat) / F.softplus(torch.tensor(1.0, device=compat.device))
        per_pair_wrong = m_penalty * occ_combined * cutoff * self._hew_confs[None, :]
        e_wrong = self.config.wrong_atom_weight * per_pair_wrong.sum()

        # E_clash
        e_clash = self._compute_clash(x) if self._protein_coords is not None else torch.tensor(0.0, device=x.device)

        # E_overfill: use occupancy Gaussian for overfill check
        total_O = O.sum(dim=0)  # per-site sum of occupancy Gaussians
        excess = F.softplus(total_O - self.config.overfill_max_per_site)
        e_overfill = self.config.overfill_weight * (excess * self._hew_confs).sum()

        e_total = global_scale * e_reward + e_wrong + e_clash + e_overfill

        # Optional logging
        if self.config.log_gradient_norms:
            self._log_step(t_val, lam_cap, lam_occ, global_scale, dist, compat, O)

        return -e_total

    def _compute_clash(self, coords: torch.Tensor) -> torch.Tensor:
        if self._protein_coords is None or self._protein_coords.numel() == 0:
            return torch.tensor(0.0, device=coords.device)
        d_min = self.config.clash_distance
        sigma = self.config.clash_sigma
        diff = coords[:, None, :] - self._protein_coords[None, :, :]
        dist_p = torch.linalg.norm(diff, dim=-1)
        clash = F.softplus(-(dist_p - d_min) / sigma) * sigma
        return self.config.clash_weight * clash.sum()

    def _log_step(self, t_val, lam_cap, lam_occ, global_scale, dist, compat, O):
        self._log.setdefault("t", []).append(t_val)
        self._log.setdefault("lam_cap", []).append(lam_cap)
        self._log.setdefault("lam_occ", []).append(lam_occ)
        self._log.setdefault("global_scale", []).append(global_scale)
        self._log.setdefault("min_dist_hew", []).append(dist.min().item())
        self._log.setdefault("mean_dist_hew", []).append(dist.mean().item())
        self._log.setdefault("max_occupancy", []).append(O.max().item())
        self._log.setdefault("mean_compat", []).append(compat.mean().item())

    # ------------------------------------------------------------------
    # Diagnostics (called after generation)
    # ------------------------------------------------------------------

    def get_log(self) -> dict[str, list[float]]:
        return dict(self._log)

    def compute_diagnostics(self, x: torch.Tensor, h: torch.Tensor) -> dict[str, Any]:
        """Post-generation diagnostics for analysis."""
        with torch.no_grad():
            n_atoms = x.shape[0]
            n_hew = self._n_hew
            if n_hew == 0:
                return {"n_candidate_hew": 0}

            rel = x[:, None, :] - self._hew_centers[None, :, :]
            dist = torch.linalg.norm(rel, dim=-1)
            C, O = self._compute_both_gaussians(dist)

            atom_probs = F.softmax(h, dim=-1)
            compat = self._compute_compatibility(atom_probs, self._hew_env_indices)

            # Direct occupancy: compatible atom within d < direct_occ_threshold
            m_reward_raw = F.softplus(compat) / F.softplus(torch.tensor(1.0, device=compat.device))
            threshold = self.config.direct_occ_threshold
            direct_mask = (dist < threshold) & (m_reward_raw > 0.1)
            n_direct = direct_mask.sum().item()

            # Soft occupancy
            soft_occ = O.max().item()

            # Nearest compatible distance
            compat_dist = dist.clone()
            compat_dist[m_reward_raw < 0.1] = float('inf')
            nearest_compat_d = compat_dist.min().item() if compat_dist.min().item() < float('inf') else float('nan')

            # Capture score
            capture_score = (C * m_reward_raw).max().item()

            # Per-site stats
            per_site_compat_d = []
            for j in range(n_hew):
                cd_j = dist[:, j].clone()
                cd_j[m_reward_raw[:, j] < 0.1] = float('inf')
                min_d = cd_j.min().item()
                if min_d < float('inf'):
                    per_site_compat_d.append(min_d)

            return {
                "n_candidate_hew": n_hew,
                "hew_environments": self._hew_envs,
                "hew_confidences": self._hew_confs.cpu().tolist(),
                "capture_score_max": capture_score,
                "soft_occ_max": soft_occ,
                "n_direct_occ": n_direct,
                "nearest_compat_d": nearest_compat_d,
                "per_site_min_compat_d": per_site_compat_d,
                "min_dist_any_atom": dist.min().item(),
                "mean_dist_any_atom": dist.mean().item(),
                "mean_compat": compat.mean().item(),
            }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def create_v6d2_guide(
    site_map: dict | str | Path,
    esfield_lambda: float = 0.5,
    protein_coords: torch.Tensor | None = None,
    **kwargs,
) -> AnalyticESFieldGuideV2:
    """Create an AnalyticESFieldGuideV2 with common defaults."""
    config_dict = {
        "esfield_lambda": esfield_lambda,
        "guidance_start": 0.3,
        "guidance_end": 0.88,
        "grad_clip": 1.0,
        "sigma_cap": 2.5,
        "sigma_occ": 1.0,
        "wrong_atom_weight": 0.5,
        "clash_weight": 0.0,
        "overfill_weight": 0.3,
        "min_confidence": 0.7,
        "top_k": 5,
        "cutoff_dist": 6.0,
        "cutoff_smooth_width": 0.5,
        "softmax_tau": 1.0,
        "direct_occ_threshold": 2.0,
    }
    config_dict.update(kwargs)
    valid_keys = [f.name for f in V6D2Config.__dataclass_fields__.values()]
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = V6D2Config(**filtered)
    return AnalyticESFieldGuideV2(site_map, config=config, protein_coords=protein_coords)
