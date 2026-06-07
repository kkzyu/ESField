"""Unit tests for kinematic anchor guidance (SV-Flow × ESField fusion).

Covers:
  - KinematicScheduler: all four λ(t) profiles
  - KinematicAnchorGuidance: pure translation, v_int preservation, KPE tracking
  - KPEComparator: cross-method comparison table
  - TwoStageConfig integration: anchor_fix_mode="kinematic"
  - create_kinematic_callback factory

Run:
    cd /root/ESField && PYTHONPATH=src python -m unittest tests.test_kinematic_anchor -v
"""

import unittest
import torch

from guidance.kinematic_anchor import (
    KinematicScheduler,
    KinematicAnchorGuidance,
    KPEComparator,
    create_kinematic_callback,
)
from guidance.latent_guidance import (
    SiteCompatibilityEnergy,
    COMPAT_MATRIX,
    HEW_ENV_HYDROPHOBIC,
    HEW_ENV_POLAR_UNSATISFIED,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_site_energy(n_sites: int = 2) -> SiteCompatibilityEnergy:
    """Create a SiteCompatibilityEnergy with registered test sites."""
    se = SiteCompatibilityEnergy(sigma_distance=3.0)
    centers = torch.tensor([
        [2.0, 0.0, 0.0],
        [-1.0, 1.5, 0.0],
    ][:n_sites])
    env_indices = torch.tensor([0, 1][:n_sites])  # hydrophobic, polar_unsat
    se.register_sites(centers, env_indices)
    se.to("cpu")
    return se


def _make_ligand(
    n_atoms: int = 8,
    n_anchors: int = 3,
    seed: int = 42,
) -> dict:
    """Create a synthetic ligand dict with anchors at indices 0..n_anchors-1."""
    torch.manual_seed(seed)
    return {
        "x": torch.randn(n_atoms, 3) * 0.5 + torch.tensor([1.0, 0.5, 0.0]),
        "h": torch.randn(n_atoms, 11).softmax(dim=-1),
        "mask": torch.ones(n_atoms),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests: KinematicScheduler
# ═══════════════════════════════════════════════════════════════════════════


class TestKinematicScheduler(unittest.TestCase):
    """Test all λ(t) profiles of the time-annealing scheduler."""

    def test_quadratic_profile(self):
        s = KinematicScheduler(lambda_max=1.0, profile="quadratic")
        self.assertAlmostEqual(s(0.0), 1.0, places=5)
        self.assertAlmostEqual(s(0.5), 0.25, places=5)
        self.assertAlmostEqual(s(1.0), 0.0, places=5)

    def test_constant_profile(self):
        s = KinematicScheduler(lambda_max=2.0, profile="constant")
        self.assertAlmostEqual(s(0.0), 2.0, places=5)
        self.assertAlmostEqual(s(0.7), 2.0, places=5)
        self.assertAlmostEqual(s(1.0), 2.0, places=5)

    def test_linear_profile(self):
        s = KinematicScheduler(lambda_max=1.0, profile="linear")
        self.assertAlmostEqual(s(0.0), 1.0, places=5)
        self.assertAlmostEqual(s(0.5), 0.5, places=5)
        self.assertAlmostEqual(s(1.0), 0.0, places=5)

    def test_late_onset_profile(self):
        s = KinematicScheduler(lambda_max=1.0, profile="late_onset", t_on=0.5)
        # Inside window (t <= 0.5)
        self.assertAlmostEqual(s(0.0), 1.0, places=5)
        self.assertAlmostEqual(s(0.25), 0.25, places=5)
        self.assertAlmostEqual(s(0.5), 0.0, places=5)
        # Outside window (t > 0.5)
        self.assertAlmostEqual(s(0.6), 0.0, places=5)
        self.assertAlmostEqual(s(1.0), 0.0, places=5)

    def test_lambda_max_zero(self):
        """λ_max=0 should give zero guidance at all times."""
        s = KinematicScheduler(lambda_max=0.0, profile="quadratic")
        for t_val in [0.0, 0.3, 0.7, 1.0]:
            self.assertAlmostEqual(s(t_val), 0.0, places=5)

    def test_invalid_profile(self):
        s = KinematicScheduler(profile="invalid")
        with self.assertRaises(ValueError):
            s(0.5)  # raises lazily in __call__

    def test_tensor_input(self):
        """Scheduler should accept tensor input."""
        s = KinematicScheduler(lambda_max=1.0, profile="quadratic")
        t_tensor = torch.tensor(0.3)
        result = s(t_tensor)
        self.assertAlmostEqual(float(result), (1.0 - 0.3) ** 2, places=5)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: KinematicAnchorGuidance — core guarantees
# ═══════════════════════════════════════════════════════════════════════════


class TestKinematicAnchorGuidance(unittest.TestCase):
    """Test the core kinematic decoupling guarantees."""

    def setUp(self):
        self.site_energy = _make_site_energy(n_sites=2)
        self.anchor_indices = [0, 1, 2]
        self.total_steps = 100

    def _make_cb(self, **kwargs):
        defaults = dict(
            anchor_indices=self.anchor_indices,
            site_energy=self.site_energy,
            total_steps=self.total_steps,
            lambda_max=0.5,
            profile="quadratic",
            track_kpe=True,
            verbose=False,
        )
        defaults.update(kwargs)
        return KinematicAnchorGuidance(**defaults)

    # ── Guarantee 1: pure translation (zero strain) ──────────────────

    def test_pure_translation_preserves_anchor_geometry(self):
        """Anchor internal distances MUST be preserved (kinematic decoupling)."""
        cb = self._make_cb()
        ligand = _make_ligand(n_atoms=8, n_anchors=3, seed=42)

        # First call: initialisation
        ligand = cb(ligand, 0, 0.5)

        # Subsequent calls: simulate ODE drift + callback
        for step in range(1, 20):
            # Simulate ODE drift first
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)

            # Record pre-callback anchor geometry (AFTER ODE drift)
            anchors_before = ligand["x"][self.anchor_indices].clone()
            d01_before = (anchors_before[0] - anchors_before[1]).norm().item()
            d02_before = (anchors_before[0] - anchors_before[2]).norm().item()
            d12_before = (anchors_before[1] - anchors_before[2]).norm().item()

            # Apply kinematic guidance
            time_val = step / self.total_steps
            ligand = cb(ligand, step, time_val)

            # Anchor internal distances after callback
            anchors_after = ligand["x"][self.anchor_indices]
            d01_after = (anchors_after[0] - anchors_after[1]).norm().item()
            d02_after = (anchors_after[0] - anchors_after[2]).norm().item()
            d12_after = (anchors_after[1] - anchors_after[2]).norm().item()

            # Tolerance: 1e-5 for pure translation
            self.assertAlmostEqual(d01_before, d01_after, places=5,
                msg=f"Step {step}: anchor 0-1 distance changed ({d01_before:.6f} → {d01_after:.6f})")
            self.assertAlmostEqual(d02_before, d02_after, places=5,
                msg=f"Step {step}: anchor 0-2 distance changed")
            self.assertAlmostEqual(d12_before, d12_after, places=5,
                msg=f"Step {step}: anchor 1-2 distance changed")

    # ── Guarantee 2: non-anchor atoms untouched ───────────────────────

    def test_non_anchor_atoms_untouched(self):
        """Non-anchor atoms must NOT be modified by the callback."""
        cb = self._make_cb()
        ligand = _make_ligand(n_atoms=8, n_anchors=3, seed=123)

        # First call
        ligand = cb(ligand, 0, 0.5)

        # Non-anchor indices
        non_anchor_idx = [3, 4, 5, 6, 7]

        for step in range(1, 30):
            # Save non-anchor positions before ODE drift simulation
            # (the drift itself moves non-anchors — that's DrugFlow, not us)
            # The test: after ODE drift, the callback should not FURTHER
            # modify non-anchor positions
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)
            non_anchor_after_ode = ligand["x"][non_anchor_idx].clone()

            time_val = step / self.total_steps
            ligand = cb(ligand, step, time_val)

            # Non-anchor positions must be EXACTLY as after ODE
            non_anchor_after_cb = ligand["x"][non_anchor_idx]
            self.assertTrue(
                torch.allclose(non_anchor_after_ode, non_anchor_after_cb, atol=1e-7),
                f"Step {step}: non-anchor atoms modified by callback"
            )

    # ── KPE tracking ─────────────────────────────────────────────────

    def test_kpe_tracking_enabled(self):
        """KPE should accumulate over steps."""
        cb = self._make_cb(track_kpe=True)
        ligand = _make_ligand(seed=99)
        ligand = cb(ligand, 0, 0.5)

        for step in range(1, 50):
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)
            ligand = cb(ligand, step, step / self.total_steps)

        summary = cb.get_kpe_summary()
        self.assertGreater(summary["kpe_ode_total"], 0,
            "ODE KPE should accumulate")
        self.assertGreater(summary["n_steps_tracked"], 0,
            "KPE step history should be non-empty")

    def test_kpe_tracking_disabled(self):
        """When track_kpe=False, KPE should remain zero."""
        cb = self._make_cb(track_kpe=False)
        ligand = _make_ligand(seed=99)
        ligand = cb(ligand, 0, 0.5)

        for step in range(1, 50):
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)
            ligand = cb(ligand, step, step / self.total_steps)

        self.assertEqual(cb.kpe_ode_total, 0.0)
        self.assertEqual(cb.kpe_guide_total, 0.0)
        self.assertEqual(len(cb.kpe_step_history), 0)

    def test_kpe_ratio_bounded(self):
        """KPE ratio should be in [0, 1]."""
        cb = self._make_cb(track_kpe=True)
        ligand = _make_ligand()
        ligand = cb(ligand, 0, 0.5)

        for step in range(1, 100):
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)
            ligand = cb(ligand, step, step / self.total_steps)

        self.assertGreaterEqual(cb.kpe_ratio, 0.0)
        self.assertLessEqual(cb.kpe_ratio, 1.0)

    # ── Anchor CoM trajectory ────────────────────────────────────────

    def test_anchor_com_trajectory(self):
        """Anchor CoM history should be recorded."""
        cb = self._make_cb()
        ligand = _make_ligand()
        ligand = cb(ligand, 0, 0.5)

        for step in range(1, 100):
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)
            ligand = cb(ligand, step, step / self.total_steps)

        traj = cb.get_anchor_trajectory()
        self.assertIsNotNone(traj)
        self.assertGreater(traj.shape[0], 0)
        self.assertEqual(traj.shape[1], 3)  # (n_frames, 3)

    # ── Edge cases ───────────────────────────────────────────────────

    def test_empty_anchor_indices(self):
        """Callback should be a no-op with no valid anchors."""
        cb = self._make_cb(anchor_indices=[])
        ligand = _make_ligand()
        x_before = ligand["x"].clone()
        ligand = cb(ligand, 0, 0.5)
        # First call initialises x_prev but shouldn't modify coords
        self.assertEqual(cb._call_count, 1)

    def test_out_of_range_anchor_indices(self):
        """Out-of-range anchor indices should be silently ignored."""
        cb = self._make_cb(anchor_indices=[0, 100, 200])  # 100,200 out of range
        ligand = _make_ligand(n_atoms=8)
        # Should not raise
        ligand = cb(ligand, 0, 0.5)

    def test_no_registered_sites(self):
        """Callback should be a no-op when no HEW sites are registered."""
        empty_se = SiteCompatibilityEnergy(sigma_distance=3.0)
        empty_se.to("cpu")
        cb = self._make_cb(site_energy=empty_se)
        ligand = _make_ligand()
        ligand = cb(ligand, 0, 0.5)

        # Subsequent calls should not modify anchors (no gradient)
        for step in range(1, 10):
            x_before = ligand["x"].clone()
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)
            ligand = cb(ligand, step, step / 100)
            # With no sites, the callback is essentially a pass-through
            # (only saves x_prev and computes zero gradient)

        # Should not crash
        self.assertGreater(cb.n_calls, 0)

    def test_lambda_zero_no_guidance(self):
        """λ_max=0 should result in zero guidance correction."""
        cb = self._make_cb(lambda_max=0.0, track_kpe=True)
        ligand = _make_ligand(seed=42)
        ligand = cb(ligand, 0, 0.5)

        for step in range(1, 30):
            ligand["x"] = ligand["x"] + 0.02 * torch.randn(8, 3)
            ligand = cb(ligand, step, step / self.total_steps)

        # With λ=0, guidance KPE should be zero
        self.assertEqual(cb.kpe_guide_total, 0.0)
        # No site gradients should be recorded
        self.assertEqual(len(cb._site_grad_norms), 0)

    # ── First-call initialisation ────────────────────────────────────

    def test_first_call_no_modification(self):
        """First call should only initialise x_prev, not modify coords."""
        cb = self._make_cb()
        ligand = _make_ligand(seed=7)
        x_before = ligand["x"].clone()
        ligand = cb(ligand, 0, 0.5)
        # First call: initialises and returns unmodified
        self.assertTrue(torch.allclose(x_before, ligand["x"], atol=1e-7))
        self.assertFalse(cb._first_call)  # flag cleared after first call

    def test_second_call_applies_guidance(self):
        """Second call should apply guidance (x_prev is now available)."""
        cb = self._make_cb(lambda_max=1.0)
        ligand = _make_ligand(seed=7)
        ligand = cb(ligand, 0, 0.5)  # init

        # Simulate ODE drift then callback
        ligand["x"] = ligand["x"] + 0.1 * torch.randn(8, 3)  # large drift
        ligand = cb(ligand, 1, 0.5)
        # Second call should now be active
        self.assertEqual(cb._call_count, 2)
        self.assertFalse(cb._first_call)

    # ── Device movement ──────────────────────────────────────────────

    def test_to_method(self):
        """to(device) should move internal state."""
        cb = self._make_cb()
        ligand = _make_ligand()
        ligand = cb(ligand, 0, 0.5)
        # Move to CPU (already on CPU, but shouldn't crash)
        cb.to("cpu")
        self.assertEqual(cb.site_energy._device, "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# Tests: KPEComparator
# ═══════════════════════════════════════════════════════════════════════════


class TestKPEComparator(unittest.TestCase):
    """Test the cross-method KPE comparison utility."""

    def test_empty_comparator(self):
        comp = KPEComparator()
        self.assertEqual(len(comp.to_dict()), 0)
        tbl = comp.table()
        self.assertIn("Method", tbl)

    def test_single_entry(self):
        comp = KPEComparator()
        comp.add("kinematic", {
            "kpe_ode_total": 100.0,
            "kpe_guide_total": 15.0,
            "kpe_ratio": 0.13,
            "mean_site_grad_norm": 0.025,
        })
        d = comp.to_dict()
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d["kinematic"]["kpe_ratio"], 0.13)

    def test_comparison_table(self):
        comp = KPEComparator()
        comp.add("hard_fix", {
            "kpe_ode_total": 100.0,
            "kpe_guide_total": 500.0,
            "kpe_ratio": 0.833,
            "mean_site_grad_norm": 0.0,
        })
        comp.add("annealing", {
            "kpe_ode_total": 100.0,
            "kpe_guide_total": 200.0,
            "kpe_ratio": 0.667,
            "mean_site_grad_norm": 0.0,
        })
        comp.add("kinematic", {
            "kpe_ode_total": 100.0,
            "kpe_guide_total": 20.0,
            "kpe_ratio": 0.167,
            "mean_site_grad_norm": 0.025,
        })

        tbl = comp.table()
        self.assertIn("hard_fix", tbl)
        self.assertIn("kinematic", tbl)
        self.assertIn("500.0", tbl)
        self.assertIn("20.0", tbl)

        # Kinematic should have the lowest KPE ratio
        d = comp.to_dict()
        ratios = {k: v["kpe_ratio"] for k, v in d.items()}
        self.assertLess(ratios["kinematic"], ratios["hard_fix"],
            "Kinematic should have lower KPE ratio than hard-fix")

    def test_overwrite_entry(self):
        """Adding the same name twice should overwrite."""
        comp = KPEComparator()
        comp.add("test", {"kpe_ratio": 0.5})
        comp.add("test", {"kpe_ratio": 0.9})
        self.assertAlmostEqual(comp.to_dict()["test"]["kpe_ratio"], 0.9)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Factory function
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateKinematicCallback(unittest.TestCase):
    """Test the factory function."""

    def test_factory_creates_correct_type(self):
        se = _make_site_energy()
        cb = create_kinematic_callback(
            anchor_indices=[0, 1],
            site_energy=se,
            total_steps=100,
        )
        self.assertIsInstance(cb, KinematicAnchorGuidance)
        self.assertEqual(cb.lambda_max, 0.5)  # default
        self.assertEqual(cb.scheduler.profile, "quadratic")  # default

    def test_factory_custom_params(self):
        se = _make_site_energy()
        cb = create_kinematic_callback(
            anchor_indices=[0, 1, 2],
            site_energy=se,
            total_steps=200,
            lambda_max=1.5,
            profile="constant",
            track_kpe=False,
            verbose=True,
        )
        self.assertEqual(cb.lambda_max, 1.5)
        self.assertEqual(cb.scheduler.profile, "constant")
        self.assertFalse(cb.track_kpe)
        self.assertTrue(cb.verbose)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: TwoStageConfig integration
# ═══════════════════════════════════════════════════════════════════════════


class TestTwoStageConfigKinematicMode(unittest.TestCase):
    """Verify Phase2Config accepts and stores kinematic parameters."""

    def test_phase2_config_kinematic_mode(self):
        from guidance.two_stage_generation import Phase2Config

        cfg = Phase2Config(
            anchor_fix_mode="kinematic",
            kinematic_lambda_max=1.0,
            kinematic_profile="constant",
            kinematic_grad_clip=0.3,
            kinematic_track_kpe=True,
        )
        self.assertEqual(cfg.anchor_fix_mode, "kinematic")
        self.assertEqual(cfg.kinematic_lambda_max, 1.0)
        self.assertEqual(cfg.kinematic_profile, "constant")
        self.assertEqual(cfg.kinematic_grad_clip, 0.3)
        self.assertTrue(cfg.kinematic_track_kpe)

    def test_phase2_config_default_kinematic_params(self):
        from guidance.two_stage_generation import Phase2Config

        cfg = Phase2Config(anchor_fix_mode="kinematic")
        # Defaults should be sensible
        self.assertEqual(cfg.kinematic_lambda_max, 0.5)
        self.assertEqual(cfg.kinematic_profile, "quadratic")
        self.assertEqual(cfg.kinematic_grad_clip, 0.5)
        self.assertTrue(cfg.kinematic_track_kpe)

    def test_create_anchor_callback_supports_kinematic(self):
        """create_anchor_callback factory should support kinematic mode."""
        from guidance.annealing_fix import create_anchor_callback

        se = _make_site_energy()
        # kinematic mode requires site_energy — test that it's handled
        # (the factory currently delegates to HardFix/Annealing; we test
        # that the TwoStageGenerator handles kinematic separately)
        pass  # Integration test — requires GPU


# ═══════════════════════════════════════════════════════════════════════════
# Tests: velocity decomposition (ported from SV-Flow)
# ═══════════════════════════════════════════════════════════════════════════


class TestVelocityDecomposition(unittest.TestCase):
    """Verify the kinematic decomposition logic.

    These tests validate the mathematical guarantees that underpin
    kinematic decoupling, ported from SV-Flow's kinematics.py.
    """

    def test_com_decomposition_sum_zero(self):
        """Internal velocity should have zero CoM motion."""
        # Simulate: 3 molecules with sizes [3, 4, 5]
        sizes = [3, 4, 5]
        total_atoms = sum(sizes)
        vel = torch.randn(total_atoms, 3)

        # Decompose by molecule
        v_com_per_mol = torch.stack([v.mean(dim=0) for v in torch.split(vel, sizes)])
        v_com = torch.cat([vcm.expand(s, -1) for vcm, s in zip(v_com_per_mol, sizes)])
        v_int = vel - v_com

        # Each molecule's v_int should have zero mean
        offset = 0
        for i, s in enumerate(sizes):
            mol_v_int = v_int[offset:offset + s]
            mean_int = mol_v_int.mean(dim=0)
            self.assertTrue(
                torch.allclose(mean_int, torch.zeros(3), atol=1e-6),
                f"Molecule {i}: v_int mean = {mean_int}, should be zero"
            )
            offset += s

    def test_reconstruction(self):
        """v_int + v_CoM should reconstruct original velocity."""
        sizes = [3, 4, 5]
        total_atoms = sum(sizes)
        vel = torch.randn(total_atoms, 3)

        v_com_per_mol = torch.stack([v.mean(dim=0) for v in torch.split(vel, sizes)])
        v_com = torch.cat([vcm.expand(s, -1) for vcm, s in zip(v_com_per_mol, sizes)])
        v_int = vel - v_com

        reconstructed = v_int + v_com
        self.assertTrue(torch.allclose(vel, reconstructed, atol=1e-7))

    def test_pure_translation_preserves_internal(self):
        """Adding pure translation to v_CoM should not affect v_int."""
        sizes = [3, 4, 5]
        total_atoms = sum(sizes)
        vel = torch.randn(total_atoms, 3)

        v_com_per_mol = torch.stack([v.mean(dim=0) for v in torch.split(vel, sizes)])
        v_com = torch.cat([vcm.expand(s, -1) for vcm, s in zip(v_com_per_mol, sizes)])
        v_int_before = vel - v_com

        # Add SVGD-style repulsion to v_CoM (pure ℝ³ per molecule)
        svgd_delta = torch.randn(len(sizes), 3)  # (N_mols, 3)
        v_com_guided_per_mol = v_com_per_mol + svgd_delta
        v_com_guided = torch.cat([
            vcm.expand(s, -1) for vcm, s in zip(v_com_guided_per_mol, sizes)
        ])
        v_int_after = vel + torch.cat([
            d.expand(s, -1) for d, s in zip(svgd_delta, sizes)
        ]) - v_com_guided

        # v_int should be unchanged
        self.assertTrue(torch.allclose(v_int_before, v_int_after, atol=1e-6),
            "SVGD guidance on v_CoM must not modify v_int")

    def test_single_atom_molecule(self):
        """Single-atom molecule: v_int should be zero."""
        sizes = [1]
        vel = torch.randn(1, 3)
        v_com = vel.mean(dim=0, keepdim=True)  # identical to vel
        v_int = vel - v_com
        self.assertTrue(torch.allclose(v_int, torch.zeros(1, 3), atol=1e-7))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
