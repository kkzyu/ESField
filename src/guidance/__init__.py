"""Inference-time guidance utilities.

Modules:
  - energy_guidance:         v4/v5 learned-potential coordinate guidance
  - flow_matching_guidance:  helper to inject guidance into flow-matching velocities
  - lambda_schedule:         time-dependent guidance strength schedules
  - latent_guidance:         v7 site-compatibility energy + Metadiffusion-style guidance
  - kinetic_trajectory_shaping: v7 KTS time-varying velocity scaling
  - two_stage_generation:    v7 two-stage (Occupy + Connect) generation orchestrator
"""

from guidance.energy_guidance import EnergyGuidance, EnergyGuidanceConfig, clip_by_norm
from guidance.flow_matching_guidance import apply_site_guidance_to_velocity
from guidance.lambda_schedule import guidance_lambda
from guidance.latent_guidance import (
    SiteCompatibilityEnergy,
    TypeGuidanceBias,
    apply_latent_guidance,
    build_site_energy_from_map,
    harmonic_restraint_energy,
    COMPAT_MATRIX,
    ATOM_TYPE_VOCAB,
    classify_hew_environment,
)
from guidance.kinetic_trajectory_shaping import (
    KTSScheduler,
    CompositeScheduler,
)
from guidance.two_stage_generation import (
    TwoStageGenerator,
    TwoStageConfig,
    Phase1Config,
    Phase2Config,
    AnchorAtoms,
    TwoStageGuideFn,
    suggest_anchor_types,
    AnchorTypeSelector,
)
