"""Small helpers for adding ESField forces to flow-matching velocities."""

from __future__ import annotations

from guidance.energy_guidance import EnergyGuidance


def apply_site_guidance_to_velocity(
    guidance: EnergyGuidance,
    base_velocity,
    coordinates,
    *,
    t: float,
    site_map,
    atom_type_indices=None,
    atom_type_probs=None,
):
    """Return guided velocity and diagnostics for one ODE step."""

    return guidance.guided_velocity(
        base_velocity,
        coordinates,
        t=t,
        site_map=site_map,
        atom_type_indices=atom_type_indices,
        atom_type_probs=atom_type_probs,
    )
