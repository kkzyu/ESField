"""MLP atom-site compatibility potential."""

from __future__ import annotations

from dataclasses import dataclass

from models.atom_features import ATOM_TYPE_VOCAB, atom_type_to_index
from models.distance_encoding import rbf_encode, require_torch
from models.site_features import SITE_TYPE_VOCAB, site_type_to_index

torch = require_torch()
nn = torch.nn


@dataclass(frozen=True)
class PotentialConfig:
    atom_embed_dim: int = 32
    site_embed_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 4
    rbf_bins: int = 16
    cutoff: float = 6.0
    energy_clip: float = 5.0


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class CompatibilityPotential(nn.Module):
    """Scalar learned compatibility energy E_phi(atom_i, site_j)."""

    def __init__(self, config: PotentialConfig | None = None) -> None:
        super().__init__()
        self.config = config or PotentialConfig()
        self.atom_embedding = nn.Embedding(len(ATOM_TYPE_VOCAB), self.config.atom_embed_dim)
        self.site_embedding = nn.Embedding(len(SITE_TYPE_VOCAB), self.config.site_embed_dim)
        input_dim = (
            self.config.atom_embed_dim
            + self.config.site_embed_dim
            + 3
            + 1
            + self.config.rbf_bins
            + 2
        )
        layers: list[nn.Module] = [
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.SiLU(),
        ]
        layers.extend(ResidualMLPBlock(self.config.hidden_dim) for _ in range(self.config.num_layers))
        layers.extend([nn.LayerNorm(self.config.hidden_dim), nn.Linear(self.config.hidden_dim, 1)])
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        atom_type_idx,
        site_type_idx,
        relative_position,
        distance,
        site_radius,
        site_confidence,
    ):
        atom_emb = self.atom_embedding(atom_type_idx.long())
        site_emb = self.site_embedding(site_type_idx.long())
        radius = site_radius.clamp_min(1.0e-4).unsqueeze(-1)
        rel_scaled = relative_position / radius
        dist_scaled = (distance / site_radius.clamp_min(1.0e-4)).unsqueeze(-1)
        rbf = rbf_encode(distance, num_bins=self.config.rbf_bins, cutoff=self.config.cutoff)
        scalar = torch.stack([site_radius, site_confidence], dim=-1)
        features = torch.cat([atom_emb, site_emb, rel_scaled, dist_scaled, rbf, scalar], dim=-1)
        energy = self.net(features).squeeze(-1)
        if self.config.energy_clip > 0:
            energy = self.config.energy_clip * torch.tanh(energy / self.config.energy_clip)
        return energy


def tensor_batch_from_pairs(pairs, *, device=None):
    device = device or torch.device("cpu")
    return {
        "atom_type_idx": torch.tensor([atom_type_to_index(pair.atom_type) for pair in pairs], dtype=torch.long, device=device),
        "site_type_idx": torch.tensor([site_type_to_index(pair.site_type) for pair in pairs], dtype=torch.long, device=device),
        "relative_position": torch.tensor([pair.relative_position for pair in pairs], dtype=torch.float32, device=device),
        "distance": torch.tensor([pair.distance for pair in pairs], dtype=torch.float32, device=device),
        "site_radius": torch.tensor([pair.site_radius for pair in pairs], dtype=torch.float32, device=device),
        "site_confidence": torch.tensor([pair.site_confidence for pair in pairs], dtype=torch.float32, device=device),
        "label": torch.tensor([pair.label for pair in pairs], dtype=torch.float32, device=device),
        "label_strength": torch.tensor([pair.label_strength for pair in pairs], dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Potential v5: hand-crafted energy shape + learned compatibility coefficients
# ---------------------------------------------------------------------------

class CompatibilityPotentialV5(nn.Module):
    """Potential v5 — hand-crafted energy shape × learned compatibility.

    E(a,s,d) = -alpha(a,s) * A(d) + beta(a,s) * R(d)

    where:
      A(d) = exp(-(d - d0)^2 / (2 * sigma_attr^2))    # attractive gaussian well
      R(d) = exp(-d / rho)                              # short-range repulsion
      alpha, beta >= 0                                  # learned compatibility coefficients

    Key properties:
      - Compatible pairs at 2-4Å:  alpha >> beta  →  E < 0  (attractive)
      - Incompatible pairs at 2-4Å: alpha ≈ 0, beta > 0  →  E ≥ 0  (no attraction)
      - Too close (<1.5Å):  R(d) dominates  →  E > 0  (steric repulsion)
      - Far (>6Å):  A(d) ≈ 0, R(d) ≈ 0  →  E ≈ 0  (no interaction)

    The MLP only learns alpha and beta — force direction is guaranteed by the
    hand-crafted shapes, preventing the "non-specific attractor" problem of v4.
    """

    def __init__(self, config: PotentialConfig | None = None) -> None:
        super().__init__()
        self.config = config or PotentialConfig()
        self.atom_embedding = nn.Embedding(len(ATOM_TYPE_VOCAB), self.config.atom_embed_dim)
        self.site_embedding = nn.Embedding(len(SITE_TYPE_VOCAB), self.config.site_embed_dim)
        input_dim = (
            self.config.atom_embed_dim
            + self.config.site_embed_dim
            + 3
            + 1
            + self.config.rbf_bins
            + 2
        )
        layers: list[nn.Module] = [
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.SiLU(),
        ]
        layers.extend(ResidualMLPBlock(self.config.hidden_dim) for _ in range(self.config.num_layers))
        # Output: (alpha, beta) — two non-negative compatibility coefficients
        layers.extend([
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, 2),
        ])
        self.net = nn.Sequential(*layers)

        # Hand-crafted energy shape parameters (not learned)
        self.d0 = 3.0          # optimal atom-site distance (Angstrom)
        self.sigma_attr = 1.5   # width of attractive gaussian well
        self.rho_rep = 0.5      # short-range repulsion decay length

    def _attractive_shape(self, distance):
        """A(d) = exp(-(d - d0)^2 / (2 * sigma^2)) * cutoff — gaussian well centered at d0."""
        gaussian = torch.exp(-(distance - self.d0) ** 2 / (2 * self.sigma_attr ** 2))
        # Smooth cutoff: cosine from 4.5Å to 6.0Å
        cutoff_start = 4.5
        cutoff_end = self.config.cutoff  # 6.0Å
        cutoff = torch.where(
            distance <= cutoff_start,
            torch.ones_like(distance),
            torch.where(
                distance >= cutoff_end,
                torch.zeros_like(distance),
                0.5 * (1.0 + torch.cos(torch.pi * (distance - cutoff_start) / (cutoff_end - cutoff_start)))
            )
        )
        return gaussian * cutoff

    def _repulsive_shape(self, distance):
        """R(d) = exp(-d / rho) — short-range exponential repulsion."""
        return torch.exp(-distance / self.rho_rep)

    def forward(
        self,
        atom_type_idx,
        site_type_idx,
        relative_position,
        distance,
        site_radius,
        site_confidence,
    ):
        """Returns energy E(a,s,d) = -alpha * A(d) + beta * R(d)."""
        atom_emb = self.atom_embedding(atom_type_idx.long())
        site_emb = self.site_embedding(site_type_idx.long())
        radius = site_radius.clamp_min(1.0e-4).unsqueeze(-1)
        rel_scaled = relative_position / radius
        dist_scaled = (distance / site_radius.clamp_min(1.0e-4)).unsqueeze(-1)
        rbf = rbf_encode(distance, num_bins=self.config.rbf_bins, cutoff=self.config.cutoff)
        scalar = torch.stack([site_radius, site_confidence], dim=-1)
        features = torch.cat([atom_emb, site_emb, rel_scaled, dist_scaled, rbf, scalar], dim=-1)
        logits = self.net(features)  # [batch, 2]
        alpha = nn.functional.softplus(logits[:, 0])  # >= 0
        beta = nn.functional.softplus(logits[:, 1])   # >= 0

        attr = self._attractive_shape(distance)
        rep = self._repulsive_shape(distance)
        energy = -alpha * attr + beta * rep

        if self.config.energy_clip > 0:
            energy = self.config.energy_clip * torch.tanh(energy / self.config.energy_clip)
        return energy

    def get_coefficients(self, atom_type_idx, site_type_idx, relative_position, distance,
                         site_radius, site_confidence):
        """Return (alpha, beta) coefficients for analysis."""
        atom_emb = self.atom_embedding(atom_type_idx.long())
        site_emb = self.site_embedding(site_type_idx.long())
        radius = site_radius.clamp_min(1.0e-4).unsqueeze(-1)
        rel_scaled = relative_position / radius
        dist_scaled = (distance / site_radius.clamp_min(1.0e-4)).unsqueeze(-1)
        rbf = rbf_encode(distance, num_bins=self.config.rbf_bins, cutoff=self.config.cutoff)
        scalar = torch.stack([site_radius, site_confidence], dim=-1)
        features = torch.cat([atom_emb, site_emb, rel_scaled, dist_scaled, rbf, scalar], dim=-1)
        logits = self.net(features)
        alpha = nn.functional.softplus(logits[:, 0])
        beta = nn.functional.softplus(logits[:, 1])
        return alpha, beta

