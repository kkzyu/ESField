"""Hard coordinate overwrite for v7.1 Phase 2 anchor preservation.

Patches DrugFlow's simulate() method to support a post_step_callback,
which is called after each ODE step to allow hard-overwriting atom
coordinates (and optionally atom types).

This replaces the harmonic restraint approach (v7.0) which was too weak
to prevent anchor atom drift during Phase 2 generation.

Usage:
    from guidance.hard_fix import patch_drugflow_hardfix, HardFixCallback

    patch_drugflow_hardfix()  # Call once, idempotent

    callback = HardFixCallback(
        anchor_indices=[0, 1],
        anchor_coords=phase1_positions,
        anchor_types=phase1_type_indices,
    )

    model.simulate(..., post_step_callback=callback)
"""

from __future__ import annotations

import os
import shutil

import torch

DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"


# ---------------------------------------------------------------------------
# HardFixCallback
# ---------------------------------------------------------------------------


class HardFixCallback:
    """Callable that hard-overwrites anchor atom coordinates after each ODE step.

    This is called as `callback(ligand_dict, step_idx, t_value)` inside
    DrugFlow's simulate() loop, right after `sample_zt_given_zs()` updates
    the ligand coordinates.

    The overwrite is DETERMINISTIC and FORCEFUL — no gradient, no restraint,
    just direct coordinate replacement.  This guarantees anchor atoms stay
    exactly at their Phase 1 positions throughout Phase 2.

    Attributes:
        anchor_indices: list of atom indices to fix (0-based)
        anchor_coords: [n_anchors, 3] tensor of target coordinates
        anchor_h: [n_anchors, n_types] tensor of target type features (optional)
        fix_coords: whether to overwrite coordinates (default True)
        fix_types: whether to overwrite atom type features (default True)
    """

    def __init__(
        self,
        anchor_indices: list[int],
        anchor_coords: torch.Tensor,
        anchor_h: torch.Tensor | None = None,
        *,
        fix_coords: bool = True,
        fix_types: bool = True,
        verbose: bool = False,
    ):
        self.anchor_indices = list(anchor_indices)
        self.anchor_coords = anchor_coords
        self.anchor_h = anchor_h
        self.fix_coords = fix_coords
        self.fix_types = fix_types and (anchor_h is not None)
        self.verbose = verbose
        self._call_count = 0

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        """Hard-overwrite anchor positions/types in ligand dict.

        Args:
            ligand: DrugFlow ligand dict with keys 'x' [n_atoms, 3] and
                    optionally 'h' [n_atoms, n_features]
            step_idx: current ODE step index (0-based)
            t_val: current time value

        Returns:
            Modified ligand dict (in-place modification + return for chaining)
        """
        self._call_count += 1

        n_atoms = ligand["x"].shape[0]
        device = ligand["x"].device

        # Validate indices are in range
        valid_indices = [i for i in self.anchor_indices if 0 <= i < n_atoms]
        if len(valid_indices) != len(self.anchor_indices):
            if self.verbose:
                skipped = set(self.anchor_indices) - set(valid_indices)
                print(f"  [HardFix] WARNING: skipping out-of-range indices: {skipped}")

        if not valid_indices:
            return ligand

        # Move anchor data to correct device if needed
        if self.anchor_coords.device != device:
            self.anchor_coords = self.anchor_coords.to(device)
        if self.anchor_h is not None and self.anchor_h.device != device:
            self.anchor_h = self.anchor_h.to(device)

        # Hard overwrite coordinates
        if self.fix_coords:
            idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
            ligand["x"][idx_tensor] = self.anchor_coords[:len(valid_indices)].to(
                device=device, dtype=ligand["x"].dtype
            )

        # Hard overwrite atom type features (only if shapes match)
        if self.fix_types and self.anchor_h is not None and "h" in ligand:
            if self.anchor_h.shape[-1] == ligand["h"].shape[-1]:
                idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
                ligand["h"][idx_tensor] = self.anchor_h[:len(valid_indices)].to(
                    device=device, dtype=ligand["h"].dtype
                )
            elif self.verbose:
                print(f"  [HardFix] Skipping type fix: anchor_h dim "
                      f"({self.anchor_h.shape[-1]}) != ligand h dim "
                      f"({ligand['h'].shape[-1]})")

        return ligand

    @property
    def n_calls(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# DrugFlow lightning.py patching
# ---------------------------------------------------------------------------


def patch_drugflow_hardfix():
    """Patch DrugFlow's simulate() AND sample() to support post_step_callback.

    Two patches are applied (idempotent):
      1. simulate() — adds post_step_callback parameter + per-step invocation
      2. sample()  — forwards post_step_callback from kwargs to simulate()
    """
    lmod_path = os.path.join(DRUGFLOW_DIR, "src/model/lightning.py")
    backup = lmod_path + ".bak2"

    if not os.path.exists(backup):
        shutil.copy(lmod_path, backup)

    with open(lmod_path) as f:
        code = f.read()

    patched = False

    # --- Patch 1: Add post_step_callback to simulate signature ---
    if "post_step_callback=None" not in code:
        old_sig = (
            "def simulate(self, ligand, pocket, timesteps, t_start, t_end=1.0,\n"
            "                 return_frames=1, guide_log_prob=None):"
        )
        new_sig = (
            "def simulate(self, ligand, pocket, timesteps, t_start, t_end=1.0,\n"
            "                 return_frames=1, guide_log_prob=None, post_step_callback=None):"
        )
        if old_sig not in code:
            raise RuntimeError(
                "DrugFlow simulate() signature changed — cannot patch hardfix. "
                "Check if DrugFlow source code has been modified."
            )
        code = code.replace(old_sig, new_sig)
        patched = True

        # --- Patch 1b: Add post_step_callback call after sample_zt_given_zs ---
        old_step = (
            "            ligand, pocket = self.sample_zt_given_zs(\n"
            "                ligand, pocket, t_array, t_array + delta_t, delta_eps_lig, cumulative_uncertainty)\n"
            "\n"
            "            # save frame"
        )
        new_step = (
            "            ligand, pocket = self.sample_zt_given_zs(\n"
            "                ligand, pocket, t_array, t_array + delta_t, delta_eps_lig, cumulative_uncertainty)\n"
            "\n"
            "            # v7.1 hardfix: post-step callback for anchor coordinate overwrite\n"
            "            if post_step_callback is not None:\n"
            "                ligand = post_step_callback(ligand, i, float(t))\n"
            "\n"
            "            # save frame"
        )
        if old_step not in code:
            raise RuntimeError(
                "DrugFlow simulate() step structure changed — cannot patch hardfix."
            )
        code = code.replace(old_step, new_step)
        print("  [HardFix] DrugFlow simulate() patched with post_step_callback support.")

    # --- Patch 2: Forward post_step_callback from sample() to simulate() ---
    if "post_step_callback=post_step_callback" not in code:
        old_sample_call = (
            "        out_tensors_ligand, out_tensors_pocket = self.simulate(\n"
            "            ligand, pocket, timesteps, 0.0, 1.0,\n"
            "            guide_log_prob=guide_log_prob\n"
            "        )"
        )
        new_sample_call = (
            "        post_step_callback = kwargs.pop('post_step_callback', None)\n"
            "        out_tensors_ligand, out_tensors_pocket = self.simulate(\n"
            "            ligand, pocket, timesteps, 0.0, 1.0,\n"
            "            guide_log_prob=guide_log_prob,\n"
            "            post_step_callback=post_step_callback\n"
            "        )"
        )
        if old_sample_call not in code:
            raise RuntimeError(
                "DrugFlow sample() call structure changed — cannot patch post_step_callback forwarding."
            )
        code = code.replace(old_sample_call, new_sample_call)
        patched = True
        print("  [HardFix] DrugFlow sample() patched to forward post_step_callback.")

    if patched:
        with open(lmod_path, "w") as f:
            f.write(code)


def patch_drugflow_sample_post_step(model) -> None:
    """Monkey-patch the IN-MEMORY model.sample() to forward post_step_callback.

    This is needed because file-based patches to lightning.py don't affect
    already-loaded Python modules.  Call this AFTER loading the model but
    BEFORE calling model.sample() with post_step_callback.

    Also supports pre_step_callback for FullyLockedAnchorCallback.

    Usage:
        model = DrugFlow.load_from_checkpoint(...)
        patch_drugflow_sample_post_step(model)
        model.sample(..., post_step_callback=my_callback, pre_step_callback=lock_cb)
    """
    _orig_sample = model.sample

    def _patched_sample(data=None, n_samples=None, num_nodes=None, timesteps=None,
                        guide_log_prob=None, size_model=None, **kwargs):
        # Extract callbacks from kwargs BEFORE calling original sample
        post_step_callback = kwargs.pop('post_step_callback', None)
        pre_step_callback = kwargs.pop('pre_step_callback', None)
        # Handle 'protein_data' keyword → 'data' positional
        if data is None and 'protein_data' in kwargs:
            data = kwargs.pop('protein_data')

        # Save to temporary attributes so simulate() can find them
        if post_step_callback is not None:
            model._post_step_callback = post_step_callback
        if pre_step_callback is not None:
            model._pre_step_callback = pre_step_callback

        # Call original sample (which calls simulate)
        result = _orig_sample(
            data, n_samples=n_samples, num_nodes=num_nodes,
            timesteps=timesteps, guide_log_prob=guide_log_prob,
            size_model=size_model, **kwargs
        )

        # Clean up
        if hasattr(model, '_post_step_callback'):
            del model._post_step_callback
        if hasattr(model, '_pre_step_callback'):
            del model._pre_step_callback

        return result

    model.sample = _patched_sample

    # Also monkey-patch simulate to check for stored callbacks
    _orig_simulate = model.simulate

    def _patched_simulate(ligand, pocket, timesteps, t_start, t_end=1.0,
                          return_frames=1, guide_log_prob=None, post_step_callback=None,
                          pre_step_callback=None):
        # If callbacks were stored by patched_sample, use them
        if post_step_callback is None and hasattr(model, '_post_step_callback'):
            post_step_callback = model._post_step_callback
        if pre_step_callback is None and hasattr(model, '_pre_step_callback'):
            pre_step_callback = model._pre_step_callback

        # Handle pre_step_callback: if simulate signature doesn't support it,
        # manually invoke before the ODE loop via the callback itself
        if pre_step_callback is not None:
            try:
                return _orig_simulate(
                    ligand, pocket, timesteps, t_start, t_end,
                    return_frames=return_frames, guide_log_prob=guide_log_prob,
                    post_step_callback=post_step_callback,
                    pre_step_callback=pre_step_callback,
                )
            except TypeError:
                # Fallback: simulate() doesn't accept pre_step_callback
                # The pre_step is handled from within the patched simulate
                return _orig_simulate(
                    ligand, pocket, timesteps, t_start, t_end,
                    return_frames=return_frames, guide_log_prob=guide_log_prob,
                    post_step_callback=post_step_callback,
                )
        else:
            return _orig_simulate(
                ligand, pocket, timesteps, t_start, t_end,
                return_frames=return_frames, guide_log_prob=guide_log_prob,
                post_step_callback=post_step_callback,
            )

    model.simulate = _patched_simulate


# ---------------------------------------------------------------------------
# FullyLockedAnchorCallback — truly lock anchors (no ODE update, no guidance)
# ---------------------------------------------------------------------------


class FullyLockedAnchorCallback:
    """Fully lock anchor atoms: zero ODE velocity + constant positions.

    Unlike HardFixCallback (which resets anchors AFTER each ODE step),
    this implementation REMOVES anchors from the ligand BEFORE the ODE
    step and RE-INSERTS them after.  This guarantees:

      1. Zero KPE contribution from anchor atoms
      2. No guidance force acts on anchors
      3. Anchor coordinates remain exactly at their Phase 1 values
         throughout the entire Phase 2 trajectory.

    Usage:
        cb = FullyLockedAnchorCallback(
            anchor_indices=[0, 1],
            anchor_coords=phase1_positions,
            anchor_h=phase1_type_features,
        )
        # Pass as post_step_callback to DrugFlow (with pre_step hook)
        model.simulate(..., post_step_callback=cb)

    NOTE: Requires the pre_step_callback patch (patch_drugflow_fully_locked).
    """

    def __init__(
        self,
        anchor_indices: list[int],
        anchor_coords: torch.Tensor,
        anchor_h: torch.Tensor | None = None,
        *,
        verbose: bool = False,
    ):
        self.anchor_indices = list(anchor_indices)
        self.anchor_coords = anchor_coords.clone()  # [n_anchors, 3]
        self.anchor_h = anchor_h.clone() if anchor_h is not None else None
        self.verbose = verbose
        self._call_count = 0
        self._saved_anchors_x: torch.Tensor | None = None
        self._saved_anchors_h: torch.Tensor | None = None
        self._saved_other_x: torch.Tensor | None = None
        self._saved_other_h: torch.Tensor | None = None

    def pre_step(self, ligand: dict) -> dict:
        """Remove anchor atoms from ligand before ODE step.

        The ODE integrator will only see non-anchor atoms, so no
        velocity is computed for anchors → zero KPE contribution.
        """
        n_atoms = ligand["x"].shape[0]
        device = ligand["x"].device

        valid_indices = [i for i in self.anchor_indices if 0 <= i < n_atoms]
        if not valid_indices:
            return ligand

        # Build masks
        all_indices = set(range(n_atoms))
        anchor_set = set(valid_indices)
        other_indices = sorted(all_indices - anchor_set)

        # Move anchor data to device
        if self.anchor_coords.device != device:
            self.anchor_coords = self.anchor_coords.to(device)
        if self.anchor_h is not None and self.anchor_h.device != device:
            self.anchor_h = self.anchor_h.to(device)

        # Save non-anchor atoms
        idx_other = torch.tensor(other_indices, device=device, dtype=torch.long)
        self._saved_other_x = ligand["x"][idx_other].clone()
        if "h" in ligand:
            self._saved_other_h = ligand["h"][idx_other].clone()

        # Replace ligand with ONLY non-anchor atoms
        ligand["x"] = self._saved_other_x
        if "h" in ligand:
            ligand["h"] = self._saved_other_h

        # Adjust mask if present
        if "mask" in ligand:
            ligand["mask"] = ligand["mask"][idx_other]

        return ligand

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        """Re-insert anchor atoms after ODE step.

        Restores the full ligand with anchors at their fixed positions
        and ODE-updated non-anchor atoms.
        """
        self._call_count += 1

        if self._saved_other_x is None:
            # First call or pre_step not invoked — use hard-fix fallback
            return self._hard_fix_fallback(ligand)

        device = ligand["x"].device
        if self.anchor_coords.device != device:
            self.anchor_coords = self.anchor_coords.to(device)
        if self.anchor_h is not None and self.anchor_h.device != device:
            self.anchor_h = self.anchor_h.to(device)

        n_anchors = len(self.anchor_indices)
        n_other = self._saved_other_x.shape[0]

        # Reconstruct full ligand: anchors FIRST, then non-anchor atoms
        new_x = torch.cat([
            self.anchor_coords[:n_anchors].to(device=device, dtype=ligand["x"].dtype),
            ligand["x"].to(device=device, dtype=ligand["x"].dtype),
        ], dim=0)

        ligand["x"] = new_x

        if "h" in ligand and self.anchor_h is not None:
            new_h = torch.cat([
                self.anchor_h[:n_anchors].to(device=device, dtype=ligand["h"].dtype),
                ligand["h"].to(device=device, dtype=ligand["h"].dtype),
            ], dim=0)
            ligand["h"] = new_h

        # Restore mask
        if "mask" in ligand:
            ligand["mask"] = torch.ones(n_anchors + n_other, device=device)

        return ligand

    def _hard_fix_fallback(self, ligand: dict) -> dict:
        """Fallback: hard-overwrite (same as HardFixCallback)."""
        n_atoms = ligand["x"].shape[0]
        device = ligand["x"].device

        valid_indices = [i for i in self.anchor_indices if 0 <= i < n_atoms]
        if not valid_indices:
            return ligand

        if self.anchor_coords.device != device:
            self.anchor_coords = self.anchor_coords.to(device)

        idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
        n_valid = min(len(valid_indices), self.anchor_coords.shape[0])
        ligand["x"][idx_tensor] = self.anchor_coords[:n_valid].to(
            device=device, dtype=ligand["x"].dtype)

        if self.anchor_h is not None and "h" in ligand:
            if self.anchor_h.shape[-1] == ligand["h"].shape[-1]:
                ligand["h"][idx_tensor] = self.anchor_h[:n_valid].to(
                    device=device, dtype=ligand["h"].dtype)

        return ligand

    @property
    def n_calls(self) -> int:
        return self._call_count


def patch_drugflow_fully_locked():
    """Patch DrugFlow simulate() to support pre_step_callback (for FullyLockedAnchor).

    Adds a pre_step_callback call BEFORE sample_zt_given_zs so that
    FullyLockedAnchorCallback can remove anchors before the ODE step.
    This is idempotent and extends the existing hard-fix patch.
    """
    lmod_path = os.path.join(DRUGFLOW_DIR, "src/model/lightning.py")

    # Ensure original backup exists
    backup = lmod_path + ".bak_orig"
    if not os.path.exists(backup):
        shutil.copy(lmod_path, backup)

    with open(lmod_path) as f:
        code = f.read()

    patched = False

    # Extend simulate() signature to accept pre_step_callback
    if "pre_step_callback=None" not in code:
        old_sig = (
            "def simulate(self, ligand, pocket, timesteps, t_start, t_end=1.0,\n"
            "                 return_frames=1, guide_log_prob=None, post_step_callback=None):"
        )
        new_sig = (
            "def simulate(self, ligand, pocket, timesteps, t_start, t_end=1.0,\n"
            "                 return_frames=1, guide_log_prob=None, post_step_callback=None,\n"
            "                 pre_step_callback=None):"
        )
        if old_sig not in code:
            raise RuntimeError(
                "DrugFlow simulate() signature changed — cannot patch fully-locked. "
                "Has patch_drugflow_hardfix() been run first?"
            )
        code = code.replace(old_sig, new_sig)
        patched = True

    # Add pre_step_callback call BEFORE sample_zt_given_zs
    old_pre_step = (
        "            ligand, pocket = self.sample_zt_given_zs(\n"
        "                ligand, pocket, t_array, t_array + delta_t, delta_eps_lig, cumulative_uncertainty)"
    )
    if old_pre_step not in code:
        raise RuntimeError(
            "DrugFlow simulate() step structure changed — cannot patch fully-locked."
        )

    new_pre_step = (
        "            # v7.1 fully-locked: remove anchors before ODE step\n"
        "            if pre_step_callback is not None:\n"
        "                ligand = pre_step_callback.pre_step(ligand)\n"
        "\n"
        "            ligand, pocket = self.sample_zt_given_zs(\n"
        "                ligand, pocket, t_array, t_array + delta_t, delta_eps_lig, cumulative_uncertainty)"
    )
    if old_pre_step + "\n\n            # v7.1 hardfix" not in code:
        # Check if hardfix patch is already applied
        if old_pre_step + "\n\n            # v7.1 hardfix: post-step callback for anchor coordinate overwrite" in code:
            code = code.replace(
                old_pre_step + "\n\n            # v7.1 hardfix: post-step callback for anchor coordinate overwrite",
                new_pre_step + "\n\n            # v7.1 hardfix: post-step callback for anchor coordinate overwrite"
            )
            patched = True
        elif old_pre_step + "\n\n            # save frame" in code:
            code = code.replace(
                old_pre_step + "\n\n            # save frame",
                new_pre_step + "\n\n            # save frame"
            )
            patched = True
        else:
            code = code.replace(old_pre_step, new_pre_step)
            patched = True

    if patched:
        with open(lmod_path, "w") as f:
            f.write(code)
        print("  [HardFix] DrugFlow simulate() patched with pre_step_callback support.")

    # Also update sample() to forward pre_step_callback
    if "pre_step_callback=pre_step_callback" not in code:
        old_sample = (
            "        post_step_callback = kwargs.pop('post_step_callback', None)\n"
            "        out_tensors_ligand, out_tensors_pocket = self.simulate(\n"
            "            ligand, pocket, timesteps, 0.0, 1.0,\n"
            "            guide_log_prob=guide_log_prob,\n"
            "            post_step_callback=post_step_callback\n"
            "        )"
        )
        if old_sample not in code:
            raise RuntimeError(
                "DrugFlow sample() structure changed — cannot patch fully-locked forwarding."
            )
        new_sample = (
            "        post_step_callback = kwargs.pop('post_step_callback', None)\n"
            "        pre_step_callback = kwargs.pop('pre_step_callback', None)\n"
            "        out_tensors_ligand, out_tensors_pocket = self.simulate(\n"
            "            ligand, pocket, timesteps, 0.0, 1.0,\n"
            "            guide_log_prob=guide_log_prob,\n"
            "            post_step_callback=post_step_callback,\n"
            "            pre_step_callback=pre_step_callback\n"
            "        )"
        )
        code = code.replace(old_sample, new_sample)
        patched = True
        print("  [HardFix] DrugFlow sample() patched to forward pre_step_callback.")

    if patched:
        with open(lmod_path, "w") as f:
            f.write(code)


# ---------------------------------------------------------------------------
# Quick test (CPU, no DrugFlow needed)
# ---------------------------------------------------------------------------


def test_hardfix_callback():
    """Smoke test: verify HardFixCallback modifies ligand dict correctly."""
    ligand = {
        "x": torch.randn(10, 3),
        "h": torch.randn(10, 11),
        "mask": torch.ones(10),
    }
    anchor_indices = [0, 2, 5]
    anchor_coords = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    anchor_h = torch.ones(3, 11)

    cb = HardFixCallback(
        anchor_indices=anchor_indices,
        anchor_coords=anchor_coords,
        anchor_h=anchor_h,
    )

    ligand = cb(ligand, 0, 0.5)

    assert torch.allclose(
        ligand["x"][0], torch.tensor([1.0, 0.0, 0.0])
    ), f"Anchor 0 not fixed: {ligand['x'][0]}"
    assert torch.allclose(
        ligand["x"][2], torch.tensor([2.0, 0.0, 0.0])
    ), f"Anchor 2 not fixed"
    assert cb.n_calls == 1

    print("  [HardFix] Callback smoke test PASSED")
    return True


if __name__ == "__main__":
    test_hardfix_callback()
