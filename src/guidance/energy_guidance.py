"""Compute ESField total site energy and coordinate gradients."""

from __future__ import annotations

from dataclasses import dataclass

from guidance.lambda_schedule import guidance_lambda
from models.atom_features import ATOM_TYPE_VOCAB
from models.distance_encoding import require_torch
from models.potential_network import CompatibilityPotential
from models.site_features import site_type_to_index
from site_detection.site_schema import SiteMap

torch = require_torch()


@dataclass(frozen=True)
class EnergyGuidanceConfig:
    lambda_max: float = 0.1
    guidance_start: float = 0.35
    guidance_end: float = 0.90
    lambda_schedule: str = "sigmoid"
    grad_clip: float = 1.0
    sigma_scale: float = 1.0
    cutoff_min: float = 4.0
    cutoff_radius_scale: float = 2.5


class EnergyGuidance:
    """Inference-time coordinate guidance from a compatibility potential."""

    def __init__(self, potential: CompatibilityPotential, config: EnergyGuidanceConfig | None = None) -> None:
        self.potential = potential
        self.config = config or EnergyGuidanceConfig()

    def total_energy(
        self,
        coordinates,
        *,
        site_map: SiteMap,
        atom_type_indices=None,
        atom_type_probs=None,
    ):
        if atom_type_indices is None and atom_type_probs is None:
            raise ValueError("provide either atom_type_indices or atom_type_probs")
        device = coordinates.device
        dtype = coordinates.dtype
        site_center, site_radius, site_confidence, site_type_idx = _site_tensors(site_map, device=device, dtype=dtype)
        if site_center.numel() == 0:
            return coordinates.new_zeros(())

        n_atoms = coordinates.shape[0]
        n_sites = site_center.shape[0]
        rel = coordinates[:, None, :] - site_center[None, :, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1.0e-8)
        sigma = (site_radius * self.config.sigma_scale).clamp_min(1.0e-4)
        cutoff = torch.maximum(
            torch.full_like(site_radius, self.config.cutoff_min),
            site_radius * self.config.cutoff_radius_scale,
        )
        weight = torch.exp(-(dist**2) / (2.0 * sigma[None, :] ** 2))
        weight = weight * torch.sigmoid(8.0 * (cutoff[None, :] - dist))
        weight = weight * site_confidence[None, :]

        flat_site_type_idx = site_type_idx.repeat(n_atoms)
        flat_rel = rel.reshape(n_atoms * n_sites, 3)
        flat_dist = dist.reshape(n_atoms * n_sites)
        flat_radius = site_radius.repeat(n_atoms)
        flat_confidence = site_confidence.repeat(n_atoms)

        if atom_type_probs is not None:
            energies_by_type = []
            vocab_size = min(atom_type_probs.shape[-1], len(ATOM_TYPE_VOCAB))
            for atom_type_index in range(vocab_size):
                flat_atom_type_idx = torch.full((n_atoms * n_sites,), atom_type_index, device=device, dtype=torch.long)
                energies_by_type.append(
                    self.potential(
                        flat_atom_type_idx,
                        flat_site_type_idx,
                        flat_rel,
                        flat_dist,
                        flat_radius,
                        flat_confidence,
                    ).reshape(n_atoms, n_sites)
                )
            stacked = torch.stack(energies_by_type, dim=-1)
            probs = atom_type_probs[:, :vocab_size].to(device=device, dtype=dtype)
            pair_energy = (stacked * probs[:, None, :]).sum(dim=-1)
        else:
            flat_atom_type_idx = atom_type_indices.to(device=device, dtype=torch.long).repeat_interleave(n_sites)
            pair_energy = self.potential(
                flat_atom_type_idx,
                flat_site_type_idx,
                flat_rel,
                flat_dist,
                flat_radius,
                flat_confidence,
            ).reshape(n_atoms, n_sites)

        return (pair_energy * weight).sum()

    def coordinate_gradient(self, coordinates, *, site_map: SiteMap, atom_type_indices=None, atom_type_probs=None):
        coords = coordinates.detach().clone().requires_grad_(True)
        energy = self.total_energy(
            coords,
            site_map=site_map,
            atom_type_indices=atom_type_indices,
            atom_type_probs=atom_type_probs,
        )
        grad = torch.autograd.grad(energy, coords, create_graph=False, retain_graph=False)[0]
        grad = clip_by_norm(grad, self.config.grad_clip)
        return energy.detach(), grad.detach()

    def guided_velocity(self, base_velocity, coordinates, *, t: float, site_map: SiteMap, atom_type_indices=None, atom_type_probs=None):
        lambda_t = guidance_lambda(
            t,
            lambda_max=self.config.lambda_max,
            guidance_start=self.config.guidance_start,
            guidance_end=self.config.guidance_end,
            schedule=self.config.lambda_schedule,
        )
        if lambda_t == 0.0:
            return base_velocity, {"lambda_t": 0.0, "energy": None, "grad_norm": 0.0}
        energy, grad = self.coordinate_gradient(
            coordinates,
            site_map=site_map,
            atom_type_indices=atom_type_indices,
            atom_type_probs=atom_type_probs,
        )
        guided = base_velocity - lambda_t * grad
        return guided, {"lambda_t": lambda_t, "energy": float(energy.cpu()), "grad_norm": float(grad.norm(dim=-1).max().cpu())}


def clip_by_norm(grad, max_norm: float):
    if max_norm <= 0:
        return grad
    norm = grad.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return grad * scale


def _site_tensors(site_map: SiteMap, *, device, dtype):
    center = torch.tensor([site.center for site in site_map.sites], device=device, dtype=dtype)
    radius = torch.tensor([site.radius for site in site_map.sites], device=device, dtype=dtype)
    confidence = torch.tensor([site.confidence for site in site_map.sites], device=device, dtype=dtype)
    site_type_idx = torch.tensor([site_type_to_index(site.site_type) for site in site_map.sites], device=device, dtype=torch.long)
    return center, radius, confidence, site_type_idx

