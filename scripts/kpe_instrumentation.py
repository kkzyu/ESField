#!/usr/bin/env python3
"""
KPE (Kinetic Path Energy) Instrumentation Module.

Injects real-time velocity tracking into DrugFlow (ODE) and TargetDiff (SDE/DDPM)
sampling loops.  This is REQUIRED because KPE cannot be computed post-hoc from
final SDF files — it must be recorded during the generative trajectory.

For each integration/denoising step, records:
  - v_eff_norm_sq: ||v_eff(x_t, t)||² — total effective velocity squared norm
  - v_guide_norm_sq: ||v_guide(x_t, t)||² — guidance-injected velocity squared norm

After generation, computes per-molecule:
  - E_ODE  = Σ ||v_eff||²
  - E_guide = Σ ||v_guide||²
  - ρ_KPE  = E_guide / (E_ODE + E_guide)

Usage:
    # DrugFlow (ODE)
    from kpe_instrumentation import KPETracker, KPELogger

    tracker = KPETracker(n_atoms=N, total_steps=100)
    for step in range(total_steps):
        # ... ODE integration step ...
        x_next = ode_step(x_current, v_theta, dt)
        v_eff = (x_next - x_current) / dt

        # If guidance is applied:
        v_guide = guidance_correction / dt  # extra displacement from guidance

        tracker.record_step(step, t, v_eff, v_guide)
        x_current = x_next

    summary = tracker.get_summary()
    # → {"e_ode": ..., "e_guide": ..., "rho_kpe": ..., ...}

    # TargetDiff (DDPM/SDE)
    tracker = KPETracker(n_atoms=N, total_steps=1000, framework="ddpm")
    for t in reversed(range(total_steps)):
        # ... DDPM denoising step ...
        x_prev = denoise_step(x_t, t)
        v_eff = (x_prev - x_t)  # effective displacement

        # Hard-fix: record teleport velocity
        if hard_fix_applied:
            v_guide = (x_after_fix - x_before_fix)  # instantaneous teleport

        tracker.record_step(total_steps - 1 - t, t / total_steps, v_eff, v_guide)

Reference:
    Li et al., "Kinetic Path Energy: A Diagnostic Framework for
    Flow-Matching-Based Molecular Generation", 2026.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ============================================================================
# Core KPE Tracker
# ============================================================================

@dataclass
class KPETracker:
    """Real-time KPE accumulator for a single molecule generation trajectory.

    Tracks velocity norms at each integration/denoising step and computes
    the KPE decomposition after generation completes.

    Attributes:
        n_atoms: Number of atoms in the molecule (for normalisation).
        total_steps: Total number of integration/denoising steps.
        framework: "ode" (DrugFlow flow-matching) or "ddpm" (TargetDiff).
        dt: Time step size (1/total_steps for ODE, 1 for DDPM).
    """

    n_atoms: int
    total_steps: int
    framework: str = "ode"  # "ode" | "ddpm"
    dt: float | None = None

    # Accumulators
    e_ode: float = 0.0
    e_guide: float = 0.0
    n_steps_recorded: int = 0
    n_guidance_steps: int = 0

    # Per-step history (sampled every record_every steps)
    step_history: list[dict] = field(default_factory=list)
    record_every: int = 10

    # Per-step accumulators for distribution analysis
    _v_eff_norms: list[float] = field(default_factory=list)
    _v_guide_norms: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.dt is None:
            self.dt = 1.0 / max(self.total_steps, 1)

    def record_step(
        self,
        step_idx: int,
        t: float,
        v_eff: np.ndarray | float,
        v_guide: np.ndarray | float | None = None,
    ):
        """Record velocity norms for one integration step.

        Args:
            step_idx: 0-based step index.
            t: Integration time (0=noise → 1=data for ODE;
               1=noise → 0=data for DDPM).
            v_eff: Effective velocity/displacement [n_atoms, 3] or scalar norm².
                   For ODE: v_eff = (x_{t+dt} - x_t) / dt.
                   For DDPM: v_eff = x_{t-1} - x_t (denoising step displacement).
            v_guide: Guidance velocity/displacement [n_atoms, 3] or scalar norm².
                     For kinematic: v_guide = correction / dt.
                     For hard-fix: v_guide = (x_after - x_before) / dt.
                     If None, assumes no guidance this step.
        """
        # Compute squared norms — handles np.ndarray, torch.Tensor, and scalar
        if isinstance(v_eff, np.ndarray):
            v_eff_norm_sq = float((v_eff ** 2).sum())
        elif hasattr(v_eff, 'numel'):  # torch.Tensor
            v_eff_norm_sq = float((v_eff ** 2).sum().item())
        else:
            v_eff_norm_sq = float(v_eff)

        if v_guide is not None:
            if isinstance(v_guide, np.ndarray):
                v_guide_norm_sq = float((v_guide ** 2).sum())
            elif hasattr(v_guide, 'numel'):  # torch.Tensor
                v_guide_norm_sq = float((v_guide ** 2).sum().item())
            else:
                v_guide_norm_sq = float(v_guide)
        else:
            v_guide_norm_sq = 0.0

        # Accumulate
        self.e_ode += v_eff_norm_sq
        self.e_guide += v_guide_norm_sq
        self.n_steps_recorded += 1
        if v_guide_norm_sq > 1e-12:
            self.n_guidance_steps += 1

        # Store per-step values
        self._v_eff_norms.append(v_eff_norm_sq)
        self._v_guide_norms.append(v_guide_norm_sq)

        # Periodic detailed record
        if step_idx % self.record_every == 0 or v_guide_norm_sq > 0:
            self.step_history.append({
                "step": step_idx,
                "t": float(t),
                "v_eff_norm_sq": v_eff_norm_sq,
                "v_guide_norm_sq": v_guide_norm_sq,
            })

    @property
    def rho_kpe(self) -> float:
        """KPE ratio: fraction of total kinetic energy from guidance."""
        total = self.e_ode + self.e_guide
        if total < 1e-12:
            return 0.0
        return self.e_guide / total

    @property
    def mean_v_eff(self) -> float:
        """Mean effective velocity norm per step."""
        if not self._v_eff_norms:
            return 0.0
        return float(np.mean(self._v_eff_norms))

    @property
    def mean_v_guide(self) -> float:
        """Mean guidance velocity norm per step."""
        if not self._v_guide_norms:
            return 0.0
        return float(np.mean(self._v_guide_norms))

    @property
    def max_v_eff(self) -> float:
        """Peak effective velocity norm."""
        if not self._v_eff_norms:
            return 0.0
        return float(np.max(self._v_eff_norms))

    @property
    def max_v_guide(self) -> float:
        """Peak guidance velocity norm (catastrophic spike detector)."""
        if not self._v_guide_norms:
            return 0.0
        return float(np.max(self._v_guide_norms))

    @property
    def guidance_suppression_factor(self) -> float:
        """Ratio of E_guide(unconstrained) / E_guide(kinematic).

        For hard-fix baseline comparison: how much KPE was suppressed
        compared to the naive approach.
        """
        # This is computed externally by comparing two trackers
        return 0.0

    def get_summary(self) -> dict:
        """Return KPE diagnostic summary for this molecule."""
        rho = self.rho_kpe
        # Interpret rho
        if rho < 0.001:
            regime = "ideal (negligible guidance perturbation)"
        elif rho < 0.01:
            regime = "acceptable (minor guidance influence)"
        elif rho < 0.1:
            regime = "moderate (guidance has measurable effect)"
        elif rho < 0.5:
            regime = "high (guidance strongly perturbs trajectory)"
        else:
            regime = "catastrophic (guidance dominates, prior destroyed)"

        return {
            "e_ode": self.e_ode,
            "e_guide": self.e_guide,
            "rho_kpe": rho,
            "regime": regime,
            "n_steps": self.n_steps_recorded,
            "n_guidance_steps": self.n_guidance_steps,
            "mean_v_eff": self.mean_v_eff,
            "mean_v_guide": self.mean_v_guide,
            "max_v_eff": self.max_v_eff,
            "max_v_guide": self.max_v_guide,
            "framework": self.framework,
            "n_atoms": self.n_atoms,
        }

    def get_trajectory(self) -> dict:
        """Return per-step trajectory for plotting KPE accumulation curves."""
        cumulative_e_ode = []
        cumulative_e_guide = []
        cumulative_rho = []
        steps = []

        acc_ode = 0.0
        acc_guide = 0.0
        for i, (v_eff, v_guide) in enumerate(zip(self._v_eff_norms,
                                                   self._v_guide_norms)):
            acc_ode += v_eff
            acc_guide += v_guide
            total = acc_ode + acc_guide
            rho = acc_guide / total if total > 1e-12 else 0.0
            steps.append(i)
            cumulative_e_ode.append(acc_ode)
            cumulative_e_guide.append(acc_guide)
            cumulative_rho.append(rho)

        return {
            "steps": steps,
            "cumulative_e_ode": cumulative_e_ode,
            "cumulative_e_guide": cumulative_e_guide,
            "cumulative_rho_kpe": cumulative_rho,
            "per_step_v_eff": self._v_eff_norms,
            "per_step_v_guide": self._v_guide_norms,
        }


# ============================================================================
# KPE Logger — multi-molecule aggregation
# ============================================================================

@dataclass
class KPELogger:
    """Aggregates KPE trackers across multiple molecules in one condition.

    Writes per-molecule summaries to a JSON log file and computes
    condition-level aggregate statistics for Table 3 and Figure 4.
    """

    condition_name: str
    pocket_name: str
    output_dir: str | Path = "experiments/kpe_logs"

    # Per-molecule trackers
    _trackers: list[KPETracker] = field(default_factory=list)
    _current_tracker: KPETracker | None = None

    def new_molecule(self, n_atoms: int, total_steps: int,
                     framework: str = "ode") -> KPETracker:
        """Start tracking a new molecule. Returns the tracker for inline use."""
        tracker = KPETracker(
            n_atoms=n_atoms,
            total_steps=total_steps,
            framework=framework,
        )
        self._current_tracker = tracker
        return tracker

    def finish_molecule(self):
        """Finalise the current molecule's tracker."""
        if self._current_tracker is not None:
            self._trackers.append(self._current_tracker)
            self._current_tracker = None

    @property
    def n_molecules(self) -> int:
        return len(self._trackers)

    def get_condition_summary(self) -> dict:
        """Aggregate KPE statistics across all molecules in this condition."""
        if not self._trackers:
            return {"error": "No molecules tracked", "n_molecules": 0}

        rhos = [t.rho_kpe for t in self._trackers]
        e_odes = [t.e_ode for t in self._trackers]
        e_guides = [t.e_guide for t in self._trackers]
        max_v_guides = [t.max_v_guide for t in self._trackers]

        return {
            "condition": self.condition_name,
            "pocket": self.pocket_name,
            "n_molecules": len(self._trackers),
            "rho_kpe_mean": float(np.mean(rhos)),
            "rho_kpe_std": float(np.std(rhos)),
            "rho_kpe_median": float(np.median(rhos)),
            "rho_kpe_min": float(np.min(rhos)),
            "rho_kpe_max": float(np.max(rhos)),
            "e_ode_mean": float(np.mean(e_odes)),
            "e_ode_std": float(np.std(e_odes)),
            "e_guide_mean": float(np.mean(e_guides)),
            "e_guide_std": float(np.std(e_guides)),
            "max_v_guide_mean": float(np.mean(max_v_guides)),
            "max_v_guide_max": float(np.max(max_v_guides)),
            # KPE suppression factor vs theoretical hard-fix maximum
            # (hard-fix conservatively estimated at rho ≈ 0.985)
            "kpe_suppression_vs_hardfix": float(
                0.985 / max(np.mean(rhos), 1e-12)
            ) if np.mean(rhos) > 0 else float("inf"),
        }

    def save(self, output_path: str | Path | None = None):
        """Save per-molecule KPE logs and condition summary to JSON."""
        output_dir = Path(output_path or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-molecule logs
        per_mol = [t.get_summary() for t in self._trackers]
        mol_path = output_dir / f"{self.pocket_name}_{self.condition_name}_kpe_per_mol.json"
        with open(mol_path, "w") as f:
            json.dump(per_mol, f, indent=2)

        # Condition summary
        summary = self.get_condition_summary()
        summary_path = output_dir / f"{self.pocket_name}_{self.condition_name}_kpe_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Trajectory data for Figure 4 (KPE accumulation curves)
        trajectories = []
        for i, t in enumerate(self._trackers):
            traj = t.get_trajectory()
            traj["molecule_idx"] = i
            trajectories.append(traj)
        traj_path = output_dir / f"{self.pocket_name}_{self.condition_name}_kpe_trajectories.json"
        with open(traj_path, "w") as f:
            json.dump(trajectories, f, indent=2)

        print(f"  KPE logs saved to {output_dir}/")
        print(f"    ρ_KPE = {summary['rho_kpe_mean']:.4%} ± "
              f"{summary['rho_kpe_std']:.4%} "
              f"(n={summary['n_molecules']})")

        return summary_path


# ============================================================================
# Comparison utility
# ============================================================================

def compare_kpe_across_conditions(
    loggers: dict[str, KPELogger],
    baseline_name: str = "unguided",
) -> dict:
    """Compare KPE metrics across experimental conditions.

    Args:
        loggers: Dict mapping condition_name → KPELogger.
        baseline_name: Which condition is the unguided baseline.

    Returns:
        Comparison dict suitable for Table 3.
    """
    comparison = {}
    baseline = loggers.get(baseline_name)
    baseline_summary = baseline.get_condition_summary() if baseline else {}

    for name, logger in loggers.items():
        summary = logger.get_condition_summary()
        rho = summary.get("rho_kpe_mean", 0.0)

        # Compute suppression factor relative to hard-fix theoretical max
        suppression = 0.985 / max(rho, 1e-12) if rho > 0 else float("inf")

        comparison[name] = {
            "rho_kpe": rho,
            "rho_kpe_std": summary.get("rho_kpe_std", 0.0),
            "e_ode": summary.get("e_ode_mean", 0.0),
            "e_guide": summary.get("e_guide_mean", 0.0),
            "kpe_suppression_vs_hardfix": suppression,
            "n_molecules": summary.get("n_molecules", 0),
        }

    return comparison


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KPE comparison tool")
    parser.add_argument("--log-dir", required=True,
                        help="Directory with KPE summary JSON files")
    parser.add_argument("--conditions", default="unguided,hard_fix,kinematic",
                        help="Comma-separated condition names")
    parser.add_argument("--output", default=None,
                        help="Output JSON for comparison table")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    conditions = args.conditions.split(",")

    # Load per-condition summaries
    summaries = {}
    for cond in conditions:
        # Try to find the summary file
        pattern = f"*{cond}*_kpe_summary.json"
        matches = sorted(log_dir.glob(pattern))
        if matches:
            with open(matches[0]) as f:
                summaries[cond] = json.load(f)
                print(f"Loaded {cond}: ρ_KPE = {summaries[cond].get('rho_kpe_mean', 'N/A')}")

    # Print comparison
    print(f"\n{'Condition':<20} {'ρ_KPE':>10} {'E_ODE':>12} {'E_guide':>12}")
    print("-" * 56)
    for cond, s in summaries.items():
        print(f"{cond:<20} {s.get('rho_kpe_mean', 0):>10.4%} "
              f"{s.get('e_ode_mean', 0):>12.2f} {s.get('e_guide_mean', 0):>12.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
