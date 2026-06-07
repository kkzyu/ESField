"""Kinetic Trajectory Shaping (KTS) scheduler for v7 two-stage generation.

Implements time-varying velocity scaling η(t) from the Kinetic Path Energy
(KPE) framework.  The KTS scheduler modulates guidance strength across the
denoising trajectory:

  - Early steps (t < τ_split):  BOOST  — amplify guidance to promote
    topological exploration (atom type changes, fragment growth).

  - Late steps (t > τ_split):   DAMP   — attenuate guidance to allow
    fine geometric refinement without over-constraining.

Reference:
  Kinetic Path Energy (KPE) + Kinetic Trajectory Shaping (KTS)
  The scheduler shapes the effective "kinetic energy" of the trajectory
  to control the balance between exploration and refinement.

Note on time convention:
  DrugFlow uses t ∈ [0, 1] where t=0 is pure noise and t=1 is data.
  The KTS scheduler follows the same convention:
    t < τ_split → early/noisy regime → boost
    t > τ_split → late/structured regime → damp
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KTSScheduler:
    """Time-varying velocity scaling for kinetic trajectory shaping.

    Provides η(t) — a multiplicative factor applied to the guided velocity
    at each denoising step:

        v'_θ = v_θ * η(t)

    or, when combined with guidance:

        v'_θ = η(t) * (v_θ - λ * ∇_x E_site)

    Formula
    -------
    For t ∈ [0, 1] (noise → data):

        if t < τ_split:
            η(t) = 1 + α₀ · (1 - t / τ_split)          [linear boost, max at t=0]
        else:
            η(t) = 1 - β₀ · (exp(k · (t - τ_split)) - 1)  [exponential damp, max at t=1]

    Parameters
    ----------
    alpha0 : float
        Early-stage boost strength.  At t=0, η = 1 + α₀.
        Larger values encourage more aggressive topological exploration
        during the noisy/chaotic regime.  Default 0.01 (mild).

    beta0 : float
        Late-stage damping strength.  At t=1,
        η = 1 - β₀ · (exp(k · (1 - τ_split)) - 1).
        Larger values suppress guidance more strongly during refinement.
        Default 0.01 (mild).

    tau_split : float
        Transition point (fraction of total time).  Before this, boost;
        after this, damp.  Default 0.6.

        Physical interpretation: τ_split separates the "topology design"
        phase (early, noisy) from the "geometry refinement" phase (late,
        structured).  The value 0.6 reflects that topology decisions are
        largely made in the first 60% of the trajectory.

    k : float
        Exponential damping stiffness.  Larger values create a sharper
        transition from neutral to damped.  Default 3.0.

    Examples
    --------
    >>> sched = KTSScheduler(alpha0=0.01, beta0=0.01, tau_split=0.6, k=3.0)
    >>> sched(0.0)   # early boost
    1.01
    >>> sched(0.6)   # transition point
    1.0
    >>> sched(1.0)   # late damping
    0.977...
    """

    alpha0: float = 0.01
    """Early boost strength. η(0) = 1 + α₀."""

    beta0: float = 0.01
    """Late damping strength."""

    tau_split: float = 0.6
    """Transition time (fraction of [0,1])."""

    k: float = 3.0
    """Exponential damping stiffness."""

    _clamp_min: float = 0.1
    """Absolute minimum η to prevent sign flips or stagnation."""

    _clamp_max: float = 5.0
    """Absolute maximum η to prevent instability."""

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if not 0.0 < self.tau_split < 1.0:
            raise ValueError(
                f"tau_split must be in (0, 1), got {self.tau_split}"
            )
        if self.alpha0 < 0:
            raise ValueError(f"alpha0 must be >= 0, got {self.alpha0}")
        if self.beta0 < 0:
            raise ValueError(f"beta0 must be >= 0, got {self.beta0}")
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")

    def __call__(self, t: float) -> float:
        """Compute η(t) for a single time point.

        Args:
            t: Current time in [0, 1] (0 = noise, 1 = data).

        Returns:
            Scaling factor η(t) ∈ [_clamp_min, _clamp_max].
        """
        t = max(0.0, min(1.0, float(t)))

        if t < self.tau_split:
            # Linear boost: η = 1 + α₀ · (1 - t/τ_split)
            # Decreases from 1+α₀ at t=0 to 1 at t=τ_split.
            eta = 1.0 + self.alpha0 * (1.0 - t / self.tau_split)
        else:
            # Exponential damp: η = 1 - β₀ · (exp(k·(t-τ_split)) - 1)
            # Decreases from 1 at t=τ_split to 1-β₀·(exp(k·(1-τ_split))-1) at t=1.
            delta = t - self.tau_split
            eta = 1.0 - self.beta0 * (math.exp(self.k * delta) - 1.0)

        return max(self._clamp_min, min(self._clamp_max, eta))

    def compute_schedule(self, t_values: list[float]) -> list[float]:
        """Compute η(t) for a sequence of time points.

        Useful for pre-computing the full schedule before a generation run.

        Args:
            t_values: List of time points in [0, 1].

        Returns:
            List of η values, same length as t_values.
        """
        return [self(t) for t in t_values]

    def compute_time_steps(
        self, n_steps: int, t_min: float = 0.0, t_max: float = 1.0
    ) -> tuple[list[float], list[float]]:
        """Generate uniformly spaced time steps and corresponding η values.

        Args:
            n_steps: Number of steps
            t_min: Starting time (default 0 = noise)
            t_max: Ending time (default 1 = data)

        Returns:
            (t_values, eta_values) — two lists of length n_steps.
        """
        t_values = [
            t_min + (t_max - t_min) * i / max(n_steps - 1, 1)
            for i in range(n_steps)
        ]
        eta_values = self.compute_schedule(t_values)
        return t_values, eta_values


# ---------------------------------------------------------------------------
# Composite scheduler: KTS × guidance lambda schedule
# ---------------------------------------------------------------------------

@dataclass
class CompositeScheduler:
    """Combine KTS time-shaping with a guidance strength schedule.

    This produces an effective guidance multiplier:

        λ_eff(t) = λ_base · λ_schedule(t) · η_kts(t)

    where:
      - λ_base is the nominal guidance strength
      - λ_schedule(t) gates guidance to specific time windows
      - η_kts(t) adds kinetic shaping (boost early, damp late)
    """

    kts: KTSScheduler = None  # type: ignore

    lambda_base: float = 0.5
    guidance_start: float = 0.1
    guidance_end: float = 0.95
    schedule_type: str = "constant"

    def __post_init__(self):
        if self.kts is None:
            self.kts = KTSScheduler()

    def __call__(self, t: float) -> float:
        """Compute effective guidance multiplier λ_eff(t).

        Args:
            t: Current time in [0, 1].

        Returns:
            Effective lambda value for this step.
        """
        # 1. Time window gating
        if t < self.guidance_start or t > self.guidance_end:
            return 0.0

        # 2. Schedule modulation within window
        span = max(self.guidance_end - self.guidance_start, 1e-8)
        x = (t - self.guidance_start) / span

        if self.schedule_type == "constant":
            lam_sched = 1.0
        elif self.schedule_type == "linear":
            lam_sched = x
        elif self.schedule_type == "cosine":
            lam_sched = 0.5 * (1.0 - math.cos(math.pi * x))
        elif self.schedule_type == "sigmoid":
            lam_sched = 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
        else:
            lam_sched = 1.0

        # 3. KTS kinetic shaping
        eta = self.kts(t)

        return self.lambda_base * lam_sched * eta
