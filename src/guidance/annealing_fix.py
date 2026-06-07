"""Flexible anchor annealing for v7.1a Phase 2 generation.

Extends the hard-fix callback with an annealing schedule that transitions
from hard coordinate overwrite (early steps) to harmonic restraint that
decays to zero (late steps).  This allows anchor atoms to relax toward
globally optimal positions once the molecular scaffold is established.

Motivation (from v7.1 final report):
  Hard anchor fixation guarantees HEW site occupancy but prevents global
  conformational optimization.  In the 3mfw pocket, occupied molecules
  showed *worse* Vina scores (p=0.067, δ=+0.47), suggesting that rigid
  anchors may strain the overall molecular geometry.

Algorithm:
  For total_t steps:
    - Steps 0 .. N_fix-1: hard overwrite (coordinates forced to anchor positions)
    - Steps N_fix .. total_t-1: harmonic restraint with k(t) linearly
      decaying from restraint_start to restraint_end

Energy during annealing phase:
  E_restraint(t) = k(t) * Σ_i ||x_i - x_i^anchor||^2

where k(t) = k_start - (k_start - k_end) * (t - N_fix) / (total_t - N_fix)

Usage:
    from guidance.annealing_fix import AnnealingAnchorFix

    cb = AnnealingAnchorFix(
        anchor_indices=[0, 1],
        anchor_coords=phase1_positions,
        anchor_h=phase1_type_probs,
        total_steps=100,
        fix_fraction=0.7,
        restraint_start=10.0,
        restraint_end=0.0,
    )

    # Pass as post_step_callback to DrugFlow's patched simulate()
    model.simulate(..., post_step_callback=cb)
"""

from __future__ import annotations

import torch


class AnnealingAnchorFix:
    """Post-step callback implementing hard-fix → harmonic annealing.

    Phase 1: hard coordinate overwrite (guarantees anchor placement).
    Phase 2: harmonic restraint with linearly decaying force constant
             (allows anchor relaxation).

    Attributes:
        anchor_indices: list[int] — atom indices to constrain (0-based)
        anchor_coords: torch.Tensor [n_anchors, 3] — target coordinates
        anchor_h: torch.Tensor or None — target atom type features
        total_steps: int — total Phase 2 ODE integration steps
        fix_fraction: float — fraction of total steps to hard-fix (0–1)
        restraint_start: float — initial harmonic force constant k (kcal/mol/Å²)
        restraint_end: float — final harmonic force constant k
        ramp: str — "linear" (default) or "exponential"
        fix_coords: bool — whether to hard-overwrite during fix phase
        fix_types: bool — whether to hard-overwrite types during fix phase
        verbose: bool — print diagnostics
    """

    def __init__(
        self,
        anchor_indices: list[int],
        anchor_coords: torch.Tensor,
        anchor_h: torch.Tensor | None = None,
        *,
        total_steps: int = 100,
        fix_fraction: float = 0.7,
        restraint_start: float = 10.0,
        restraint_end: float = 0.0,
        ramp: str = "linear",
        fix_coords: bool = True,
        fix_types: bool = True,
        verbose: bool = False,
    ):
        if not (0.0 <= fix_fraction <= 1.0):
            raise ValueError(f"fix_fraction must be in [0, 1], got {fix_fraction}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")

        self.anchor_indices = list(anchor_indices)
        self.anchor_coords = anchor_coords
        self.anchor_h = anchor_h
        self.total_steps = total_steps
        self.fix_fraction = fix_fraction
        self.restraint_start = restraint_start
        self.restraint_end = restraint_end
        self.ramp = ramp
        self.fix_coords = fix_coords
        self.fix_types = fix_types and (anchor_h is not None)
        self.verbose = verbose

        # Derived: N_fix = first N_fix steps use hard overwrite
        self._n_fix = int(total_steps * fix_fraction)

        # State tracking
        self._call_count = 0
        self._current_k: float = 0.0
        self._transition_step: int | None = None

        if self.verbose:
            print(
                f"  [AnnealingFix] total_steps={total_steps}, "
                f"n_fix={self._n_fix} ({fix_fraction:.0%}), "
                f"k: {restraint_start:.2f} → {restraint_end:.2f} "
                f"({ramp})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        """Apply anchor constraint for the current step.

        Args:
            ligand: DrugFlow ligand dict with 'x' [n_atoms, 3] and
                    optionally 'h' [n_atoms, n_features]
            step_idx: current ODE step (0-based)
            t_val: current time value (unused; we use step_idx)

        Returns:
            Modified ligand dict.
        """
        self._call_count += 1

        n_atoms = ligand["x"].shape[0]
        device = ligand["x"].device

        valid_indices = [i for i in self.anchor_indices if 0 <= i < n_atoms]
        if not valid_indices:
            if self.verbose and self._call_count == 1:
                print(f"  [AnnealingFix] WARNING: no valid anchor indices in "
                      f"range 0..{n_atoms - 1}")
            return ligand

        # Move anchor data to device if needed
        if self.anchor_coords.device != device:
            self.anchor_coords = self.anchor_coords.to(device)
        if self.anchor_h is not None and self.anchor_h.device != device:
            self.anchor_h = self.anchor_h.to(device)

        # Determine the constraint for this step
        n_indices = len(valid_indices)
        idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
        anchor_subset = self.anchor_coords[:n_indices].to(
            device=device, dtype=ligand["x"].dtype
        )

        if step_idx < self._n_fix:
            # ── Hard-overwrite phase ──
            if self.fix_coords:
                ligand["x"][idx_tensor] = anchor_subset
            if self.fix_types and self.anchor_h is not None and "h" in ligand:
                if self.anchor_h.shape[-1] == ligand["h"].shape[-1]:
                    ligand["h"][idx_tensor] = self.anchor_h[:n_indices].to(
                        device=device, dtype=ligand["h"].dtype
                    )
            self._current_k = float("inf")  # hard fix

            if self.verbose and step_idx == self._n_fix - 1:
                print(f"  [AnnealingFix] step {step_idx + 1}/{self.total_steps}: "
                      f"last hard-fix step, transitioning to harmonic")

        else:
            # ── Harmonic restraint phase ──
            k = self._compute_k(step_idx)
            self._current_k = k

            if k > 0 and self.fix_coords:
                # Apply harmonic restraint as a coordinate correction.
                # We add a small correction proportional to the distance
                # from target, scaled by k.  This is equivalent to adding
                # -∇E_restraint to the velocity, where
                #   E_restraint = k * Σ||x - x_target||^2
                #   -∇E_restraint = -2k * (x - x_target)
                #
                # Since DrugFlow uses a post_step_callback (not gradient),
                # we modify x directly: x ← x - α * 2k * (x - x_target)
                # where α is a step-size-adaptive factor.

                current = ligand["x"][idx_tensor]
                displacement = current - anchor_subset

                # Clip displacement for numerical stability
                max_displacement = 0.5  # Å
                displacement = torch.clamp(displacement, -max_displacement, max_displacement)

                # Correction: x_new = x - step * 2k * (x - x_target)
                # Effective step ~ 0.01 (ODE step size)
                effective_step = 0.01
                correction = effective_step * 2.0 * k * displacement
                ligand["x"][idx_tensor] = current - correction

            if (
                self.fix_types
                and self.anchor_h is not None
                and "h" in ligand
                and self.anchor_h.shape[-1] == ligand["h"].shape[-1]
            ):
                # During annealing phase, still apply soft type preservation
                # (not hard overwrite — types can adjust)
                pass  # Type preservation handled by TwoStageGuideFn

            if self.verbose and step_idx == self._n_fix:
                print(f"  [AnnealingFix] step {step_idx + 1}/{self.total_steps}: "
                      f"transitioning to harmonic, k={k:.3f}")

        return ligand

    def _compute_k(self, step_idx: int) -> float:
        """Compute the harmonic force constant for the current step."""
        if self._n_fix >= self.total_steps:
            return 0.0  # all steps are hard fix

        # Fraction through the annealing phase
        n_anneal = self.total_steps - self._n_fix
        progress = (step_idx - self._n_fix) / max(n_anneal, 1)
        progress = max(0.0, min(1.0, progress))

        if self.ramp == "linear":
            k = self.restraint_start + (self.restraint_end - self.restraint_start) * progress
        elif self.ramp == "exponential":
            # k(t) = k_start * (k_end / k_start)^(progress)
            if self.restraint_start > 0 and self.restraint_end > 0:
                ratio = self.restraint_end / self.restraint_start
                k = self.restraint_start * (ratio ** progress)
            else:
                # Fallback to linear if endpoint is zero
                k = self.restraint_start * (1.0 - progress)
        else:
            raise ValueError(f"Unknown ramp type: {self.ramp}")

        return max(0.0, k)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_calls(self) -> int:
        return self._call_count

    @property
    def n_fix_steps(self) -> int:
        return self._n_fix

    @property
    def n_anneal_steps(self) -> int:
        return max(0, self.total_steps - self._n_fix)

    @property
    def current_k(self) -> float:
        return self._current_k

    @property
    def in_fix_phase(self) -> bool:
        return self._call_count > 0 and self._call_count <= self._n_fix

    def get_schedule(self) -> list[float]:
        """Return the full k(t) schedule for all steps (for plotting)."""
        schedule = []
        for step in range(self.total_steps):
            if step < self._n_fix:
                schedule.append(float("inf"))
            else:
                schedule.append(self._compute_k(step))
        return schedule

    def to(self, device: str) -> "AnnealingAnchorFix":
        """Move tensors to the specified device."""
        self.anchor_coords = self.anchor_coords.to(device)
        if self.anchor_h is not None:
            self.anchor_h = self.anchor_h.to(device)
        return self


# ---------------------------------------------------------------------------
# Factory function for creating the appropriate callback from config
# ---------------------------------------------------------------------------


def create_anchor_callback(
    anchor_indices: list[int],
    anchor_coords: torch.Tensor,
    anchor_h: torch.Tensor | None = None,
    *,
    mode: str = "hard",
    config: dict | None = None,
    total_steps: int = 100,
    verbose: bool = False,
) -> "AnnealingAnchorFix | HardFixCallback":
    """Factory: create either HardFixCallback or AnnealingAnchorFix.

    Args:
        anchor_indices: indices of anchor atoms in the full molecule
        anchor_coords: [n_anchors, 3] target coordinates
        anchor_h: [n_anchors, n_types] target type features (optional)
        mode: "hard" (v7.1 default) or "annealing" (v7.1a)
        config: additional config dict (e.g., from yaml).  Keys used:
            - fix_fraction (float, default 0.7)
            - restraint_start (float, default 10.0)
            - restraint_end (float, default 0.0)
            - ramp (str, default "linear")
        total_steps: number of Phase 2 ODE integration steps
        verbose: print diagnostic info

    Returns:
        A callable suitable for use as DrugFlow's post_step_callback.
    """
    if mode == "hard":
        from guidance.hard_fix import HardFixCallback
        return HardFixCallback(
            anchor_indices=anchor_indices,
            anchor_coords=anchor_coords,
            anchor_h=anchor_h,
            fix_coords=True,
            fix_types=(anchor_h is not None),
            verbose=verbose,
        )

    elif mode == "annealing":
        cfg = config or {}
        return AnnealingAnchorFix(
            anchor_indices=anchor_indices,
            anchor_coords=anchor_coords,
            anchor_h=anchor_h,
            total_steps=total_steps,
            fix_fraction=cfg.get("fix_fraction", 0.7),
            restraint_start=cfg.get("restraint_start", 10.0),
            restraint_end=cfg.get("restraint_end", 0.0),
            ramp=cfg.get("ramp", "linear"),
            fix_coords=True,
            fix_types=(anchor_h is not None),
            verbose=verbose,
        )

    else:
        raise ValueError(
            f"Unknown anchor_fix_mode: {mode!r}. "
            f"Choose 'hard' or 'annealing'."
        )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_annealing_hard_fix_phase():
    """Verify hard-fix phase: coordinates are exactly overwritten."""
    ligand = {
        "x": torch.randn(10, 3),
        "h": torch.randn(10, 11),
    }
    anchor_indices = [0, 3, 7]
    anchor_coords = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    cb = AnnealingAnchorFix(
        anchor_indices=anchor_indices,
        anchor_coords=anchor_coords,
        total_steps=100,
        fix_fraction=0.7,
    )

    # Step 0 (hard fix)
    ligand = cb(ligand, 0, 0.0)
    assert torch.allclose(ligand["x"][0], anchor_coords[0], atol=1e-6), \
        f"Anchor 0 not hard-fixed: {ligand['x'][0]}"
    assert torch.allclose(ligand["x"][3], anchor_coords[1], atol=1e-6), \
        f"Anchor 3 not hard-fixed"
    assert cb.in_fix_phase
    print("  [test_annealing] Hard-fix phase: PASSED")

    # Step 69 (last hard fix)
    for s in range(1, 70):
        ligand["x"] = torch.randn(10, 3)  # scramble
        ligand = cb(ligand, s, 0.0)
    assert torch.allclose(ligand["x"][7], anchor_coords[2], atol=1e-6)
    print("  [test_annealing] Last hard-fix step: PASSED")

    # Step 70 (first harmonic) — simulate ODE drift before callback
    # In DrugFlow's simulate(), the ODE step modifies coordinates slightly
    # before the post_step_callback is invoked.  We simulate that drift.
    ligand["x"] = ligand["x"] + 0.05 * torch.randn(10, 3)  # simulate ODE drift
    original_x = ligand["x"].clone()
    ligand = cb(ligand, 70, 0.0)
    # Should have moved (soft correction, not hard overwrite)
    assert not torch.allclose(ligand["x"][0], original_x[0], atol=1e-6), \
        "Harmonic phase should modify coordinates"
    assert abs(cb.current_k - 10.0) < 0.01, \
        f"k should be ~{10.0}, got {cb.current_k}"
    print("  [test_annealing] Harmonic phase entry: PASSED")

    # Step 99 (last harmonic, k → 0)
    for s in range(71, 100):
        ligand["x"] = ligand["x"] + 0.05 * torch.randn(10, 3)  # simulate ODE drift
        ligand = cb(ligand, s, 0.0)
    # At step 99 (last step, k close to 0 but not exactly due to discrete steps)
    # With 100 steps, 70 fix + 30 anneal: at step 99, progress=29/30
    # linear k = 10.0 * (1 - 29/30) ≈ 0.33
    assert cb.current_k < 1.0, \
        f"k should be approaching 0, got {cb.current_k}"
    print("  [test_annealing] Harmonic phase completion: PASSED")


def test_annealing_exponential_ramp():
    """Verify exponential decay ramp."""
    cb = AnnealingAnchorFix(
        anchor_indices=[0],
        anchor_coords=torch.zeros(1, 3),
        total_steps=100,
        fix_fraction=0.5,
        restraint_start=10.0,
        restraint_end=1.0,
        ramp="exponential",
    )

    # Check k at midpoint of annealing phase
    k_half = cb._compute_k(75)  # (75 - 50) / 50 = 0.5 progress
    expected = 10.0 * ((1.0 / 10.0) ** 0.5)  # ~3.16
    assert abs(k_half - expected) < 0.5, f"k at 50%: {k_half} vs {expected}"
    print("  [test_annealing] Exponential ramp: PASSED")


def test_create_anchor_callback_factory():
    """Verify factory creates correct type based on mode string."""
    indices = [0, 1]
    coords = torch.randn(2, 3)

    cb_hard = create_anchor_callback(indices, coords, mode="hard")
    assert cb_hard.__class__.__name__ == "HardFixCallback", \
        f"Expected HardFixCallback, got {cb_hard.__class__.__name__}"

    cb_anneal = create_anchor_callback(indices, coords, mode="annealing",
                                       config={"fix_fraction": 0.6},
                                       total_steps=100)
    assert cb_anneal.__class__.__name__ == "AnnealingAnchorFix", \
        f"Expected AnnealingAnchorFix, got {cb_anneal.__class__.__name__}"
    assert cb_anneal._n_fix == 60  # 0.6 * 100
    print("  [test_annealing] Factory function: PASSED")


if __name__ == "__main__":
    test_annealing_hard_fix_phase()
    test_annealing_exponential_ramp()
    test_create_anchor_callback_factory()
    print("\n  All annealing tests PASSED ✓")
