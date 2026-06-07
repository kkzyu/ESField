"""Unit tests for Analytic ESField v6-D module.

Tests:
  1. Compatible atom moves toward HEW center
  2. Incompatible atom receives no displacement reward
  3. Wrong-type atom near HEW is penalized
  4. Atom too close to protein is repelled
  5. Random compatibility matrix produces different behavior
  6. All terms are differentiable
  7. HEW environment classification
  8. Actionable HEW filtering
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_site_map(sites):
    return {"sites": sites, "pocket_center": [0.0, 0.0, 0.0]}


def _hew_site(center, env="hydrophobic", confidence=0.9, radius=1.4,
              hbond_count=0, hydrophobic_contact_count=4, nearest_protein_distance=4.0):
    return {
        "site_type": "high_energy_water",
        "center": list(center),
        "radius": radius,
        "confidence": confidence,
        "features": {
            "hbond_count": hbond_count,
            "hydrophobic_contact_count": hydrophobic_contact_count,
            "nearest_protein_distance": nearest_protein_distance,
        },
    }


class TestV6DCompatibility(unittest.TestCase):
    """Test that compatible atoms are rewarded and incompatible are not."""

    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.skipTest("PyTorch not installed")

    def test_compatible_atom_moves_toward_hew(self):
        """Single HEW + C_sp3 should receive negative energy gradient toward HEW."""
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site(center=(0.0, 0.0, 0.0), env="hydrophobic", confidence=0.9)
        ])
        config = V6DConfig(disp_weight=1.0, wrong_atom_weight=0.0,
                           clash_weight=0.0, overfill_weight=0.0,
                           sigma_occ=1.2, cutoff_dist=5.0)
        guide = AnalyticESFieldGuide(site_map, config=config)
        guide.to(self.torch.device("cpu"))

        # C_sp3 atom at (2.0, 0, 0) — 2Å from HEW center
        x = self.torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        # Atom type: C_sp3 (index 1) with high probability
        h = self.torch.zeros(1, 11)
        h[0, 1] = 5.0  # strong C_sp3 signal

        energy = guide(self.torch.tensor(0.5), x=x, h=h, batch_mask=None)
        grad = self.torch.autograd.grad(energy, x)[0]

        # Energy should be finite
        self.assertTrue(self.torch.isfinite(energy).all(),
                        f"Energy not finite: {energy}")

        # Gradient should point toward HEW center (negative x direction for atom at +2.0)
        # Energy is -E_guide, so gradient of energy w.r.t x = -dE_guide/dx
        # E_disp < 0 (attractive), so -E_disp > 0
        # But guide returns -E_total for log_prob, and gradient of that moves atoms
        # toward lower E_guide (more negative).
        # For atom at (2,0,0) and HEW at (0,0,0):
        # E_disp = -B * M * exp(-d^2/(2*sigma^2))
        # dE_disp/dx = B*M * exp(-d^2/(2*sigma^2)) * x/sigma^2
        # This is positive (pushes atom away from minimum)... wait no.
        # E_disp = -B*M*exp(-4/(2*1.44)) ≈ -B*M*0.25 → negative
        # dE_disp/dx = B*M * d/dx[-exp(-x^2/(2s^2))] = B*M * exp(-x^2/(2s^2)) * x/s^2
        # = B*M * 0.25 * 2/1.44 ≈ B*M*0.35 > 0 (away from HEW? no...)
        # Actually E_disp has a minimum at d=0: E_disp = -const * exp(-d^2/2s^2)
        # dE_disp/dd = const * d/s^2 * exp(-d^2/2s^2) > 0 for d > 0
        # So gradient points AWAY from HEW center (increasing d increases E_disp toward 0)
        # Energy in guide = -E_total, grad(-E_total) = -grad(E_total)
        # grad(E_disp) pushes atom UP the energy = away from HEW
        # But grad(-E_disp) = -grad(E_disp) pushes toward HEW ✓
        # So the guide gradient should have negative x component (toward origin)

        self.assertLess(grad[0, 0].item(), 0.0,
                        f"C_sp3 at x=2.0 should be pulled toward HEW at origin, "
                        f"but grad_x = {grad[0, 0].item():.4f}")

    def test_incompatible_atom_no_displacement_reward(self):
        """Charged atom near hydrophobic HEW should get no reward."""
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site(center=(0.0, 0.0, 0.0), env="hydrophobic", confidence=0.9)
        ])
        config = V6DConfig(disp_weight=1.0, wrong_atom_weight=0.5,
                           clash_weight=0.0, overfill_weight=0.0,
                           sigma_occ=1.2, cutoff_dist=5.0)
        guide = AnalyticESFieldGuide(site_map, config=config)
        guide.to(self.torch.device("cpu"))

        # Compatible C_sp3 atom
        x_compat = self.torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        h_compat = self.torch.zeros(1, 11)
        h_compat[0, 1] = 5.0  # C_sp3

        energy_compat = guide(self.torch.tensor(0.5), x=x_compat, h=h_compat, batch_mask=None)
        grad_compat = self.torch.autograd.grad(energy_compat, x_compat)[0]
        force_compat = grad_compat.norm().item()

        # Incompatible charged atom
        x_incompat = self.torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        h_incompat = self.torch.zeros(1, 11)
        h_incompat[0, 9] = 5.0  # charged

        energy_incompat = guide(self.torch.tensor(0.5), x=x_incompat, h=h_incompat, batch_mask=None)
        grad_incompat = self.torch.autograd.grad(energy_incompat, x_incompat)[0]
        force_incompat = grad_incompat.norm().item()

        # The compatible atom should experience a stronger attractive force
        # (more negative energy → larger magnitude gradient toward HEW)
        self.assertGreater(
            force_compat, 0.0,
            f"Compatible atom should experience force, got {force_compat:.6f}"
        )

        # Incompatible atom may experience penalty force (away from HEW)
        # Key check: the two forces are different (random matrix would give same irrelevant force)
        self.assertNotEqual(
            round(force_compat, 4), round(force_incompat, 4),
            f"Compatible ({force_compat:.6f}) and incompatible ({force_incompat:.6f}) "
            f"should have different forces"
        )

    def test_wrong_atom_penalized_near_hew(self):
        """Wrong-type atom near HEW should receive positive energy penalty."""
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site(center=(0.0, 0.0, 0.0), env="hydrophobic", confidence=0.9)
        ])
        config = V6DConfig(disp_weight=0.0, wrong_atom_weight=1.0,
                           clash_weight=0.0, overfill_weight=0.0,
                           sigma_occ=1.2, cutoff_dist=5.0)
        guide = AnalyticESFieldGuide(site_map, config=config)
        guide.to(self.torch.device("cpu"))

        # Charged atom near hydrophobic HEW — should be penalized
        x = self.torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        h = self.torch.zeros(1, 11)
        h[0, 9] = 5.0  # charged

        energy = guide(self.torch.tensor(0.5), x=x, h=h, batch_mask=None)

        # Wrong-atom energy is penalized: E_wrong > 0, so -E_wrong < 0
        # The output is -E_total = -(E_disp + E_wrong)
        # With disp_weight=0, E_disp = 0
        # E_wrong > 0 for incompatible atom near HEW
        # So energy = -E_wrong < 0
        self.assertLess(
            energy.item(), 0.0,
            f"Wrong atom penalty should make total energy negative, got {energy.item():.4f}"
        )

    def test_clash_repulsion(self):
        """Atom too close to protein should be repelled."""
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site(center=(0.0, 0.0, 0.0), env="hydrophobic", confidence=0.9)
        ])
        # Protein atom at origin — ligand atom too close
        protein_coords = self.torch.tensor([[0.0, 0.0, 0.0]])

        config = V6DConfig(disp_weight=0.0, wrong_atom_weight=0.0,
                           clash_weight=1.0, overfill_weight=0.0,
                           clash_distance=2.0, clash_sigma=0.3)
        guide = AnalyticESFieldGuide(site_map, config=config, protein_coords=protein_coords)
        guide.to(self.torch.device("cpu"))

        # Ligand atom only 1.0Å from protein → should be repelled
        x = self.torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        h = self.torch.zeros(1, 11)
        h[0, 1] = 5.0  # C_sp3

        energy = guide(self.torch.tensor(0.5), x=x, h=h, batch_mask=None)
        grad = self.torch.autograd.grad(energy, x)[0]

        # 1.0Å < clash_distance (2.0Å) → clash penalty active
        # E_clash > 0, so -E_clash < 0 → energy is negative
        self.assertTrue(self.torch.isfinite(energy).all())

        # Gradient should push atom AWAY from protein (positive x direction)
        # Because E_clash increases as d decreases
        # grad(E_clash) points toward protein (decreasing d increases E_clash)
        # grad(-E_clash) = -grad(E_clash) points away from protein
        self.assertGreater(
            grad[0, 0].item(), 0.0,
            f"Atom at 1.0Å from protein should be repelled, "
            f"but grad_x = {grad[0, 0].item():.4f}"
        )

        # Atom at 5.0Å from protein → no clash (well beyond clash_distance=2.0)
        x_far = self.torch.tensor([[5.0, 0.0, 0.0]], requires_grad=True)
        energy_far = guide(self.torch.tensor(0.5), x=x_far, h=h, batch_mask=None)
        grad_far = self.torch.autograd.grad(energy_far, x_far)[0]

        # Force should be negligible at 5Å
        self.assertLess(
            grad_far.norm().item(), 0.01,
            f"Atom at 5.0Å should have negligible clash force, "
            f"got {grad_far.norm().item():.6f}"
        )

    def test_random_matrix_different_behavior(self):
        """Random compatibility matrix should produce different forces."""
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site(center=(0.0, 0.0, 0.0), env="hydrophobic", confidence=0.9)
        ])
        config = V6DConfig(disp_weight=1.0, wrong_atom_weight=0.0,
                           clash_weight=0.0, overfill_weight=0.0,
                           sigma_occ=1.2, cutoff_dist=5.0)
        guide = AnalyticESFieldGuide(site_map, config=config)
        guide.to(self.torch.device("cpu"))

        # Save the real compatibility matrix and replace with random
        real_compat = guide._compat.clone()
        guide._compat = self.torch.randn_like(real_compat)

        # C_sp3 atom
        x = self.torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        h = self.torch.zeros(1, 11)
        h[0, 1] = 5.0  # C_sp3

        energy_random = guide(self.torch.tensor(0.5), x=x, h=h, batch_mask=None)
        grad_random = self.torch.autograd.grad(energy_random, x)[0]

        # Restore real matrix
        guide._compat = real_compat
        energy_real = guide(self.torch.tensor(0.5), x=x, h=h, batch_mask=None)
        grad_real = self.torch.autograd.grad(energy_real, x)[0]

        # Forces should be different
        self.assertNotEqual(
            round(grad_random[0, 0].item(), 4),
            round(grad_real[0, 0].item(), 4),
            f"Random matrix ({grad_random[0, 0].item():.4f}) and real matrix "
            f"({grad_real[0, 0].item():.4f}) should produce different forces"
        )

    def test_all_terms_differentiable(self):
        """All energy terms must be differentiable w.r.t. coordinates."""
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site(center=(0.0, 0.0, 0.0), env="hydrophobic", confidence=0.9),
            _hew_site(center=(3.0, 0.0, 0.0), env="polar_unsatisfied", confidence=0.8,
                      hbond_count=0, hydrophobic_contact_count=1),
        ])
        protein_coords = self.torch.tensor([[0.0, 1.5, 0.0]])

        config = V6DConfig(
            disp_weight=1.0, wrong_atom_weight=0.5,
            clash_weight=1.0, overfill_weight=0.3,
            sigma_occ=1.2, cutoff_dist=5.0,
        )
        guide = AnalyticESFieldGuide(site_map, config=config, protein_coords=protein_coords)
        guide.to(self.torch.device("cpu"))

        # Multiple atoms of different types
        x = self.torch.tensor([
            [1.5, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ], requires_grad=True)
        h = self.torch.zeros(3, 11)
        h[0, 1] = 3.0  # C_sp3 at atom 0
        h[0, 5] = 1.0  # some O_acceptor
        h[1, 5] = 4.0  # O_acceptor at atom 1
        h[2, 9] = 4.0  # charged at atom 2

        energy = guide(self.torch.tensor(0.6), x=x, h=h, batch_mask=None)
        grad = self.torch.autograd.grad(energy, x, create_graph=True)[0]

        self.assertTrue(self.torch.isfinite(energy).all(),
                        f"Energy not finite: {energy}")
        self.assertTrue(self.torch.isfinite(grad).all(),
                        f"Gradient not finite: {grad}")
        self.assertEqual(grad.shape, x.shape)

        # Verify second-order differentiability: grad itself supports further grad
        grad_norm_sq = grad.pow(2).sum()
        hessian_vp = self.torch.autograd.grad(grad_norm_sq, x, retain_graph=True)[0]
        self.assertTrue(self.torch.isfinite(hessian_vp).all(),
                        f"Hessian-vector product not finite")
        self.assertEqual(hessian_vp.shape, x.shape)


class TestV6DEnvironmentClassification(unittest.TestCase):
    """Test HEW environment classification."""

    def test_classify_hydrophobic(self):
        from models.analytic_esfield import classify_hew_environment, HEW_ENV_HYDROPHOBIC

        site = _hew_site((0, 0, 0), hbond_count=0, hydrophobic_contact_count=5,
                         nearest_protein_distance=4.0)
        self.assertEqual(classify_hew_environment(site), HEW_ENV_HYDROPHOBIC)

    def test_classify_polar_unsatisfied(self):
        from models.analytic_esfield import classify_hew_environment, HEW_ENV_POLAR_UNSATISFIED

        site = _hew_site((0, 0, 0), hbond_count=0, hydrophobic_contact_count=1,
                         nearest_protein_distance=4.0)
        self.assertEqual(classify_hew_environment(site), HEW_ENV_POLAR_UNSATISFIED)

    def test_classify_buried(self):
        from models.analytic_esfield import classify_hew_environment, HEW_ENV_BURIED

        site = _hew_site((0, 0, 0), hbond_count=2, hydrophobic_contact_count=3,
                         nearest_protein_distance=2.0)
        self.assertEqual(classify_hew_environment(site), HEW_ENV_BURIED)

    def test_classify_mixed(self):
        from models.analytic_esfield import classify_hew_environment, HEW_ENV_MIXED

        site = _hew_site((0, 0, 0), hbond_count=1, hydrophobic_contact_count=3,
                         nearest_protein_distance=3.5)
        self.assertEqual(classify_hew_environment(site), HEW_ENV_MIXED)


class TestV6DFiltering(unittest.TestCase):
    """Test actionable HEW filtering."""

    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.skipTest("PyTorch not installed")

    def test_confidence_filter(self):
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site((0, 0, 0), confidence=0.9),
            _hew_site((1, 0, 0), confidence=0.2),  # below threshold
        ])
        config = V6DConfig(min_confidence=0.3)
        guide = AnalyticESFieldGuide(site_map, config=config)

        self.assertEqual(guide._n_hew, 1,
                         f"Low-confidence HEW should be filtered, got {guide._n_hew}")

    def test_top_k_selection(self):
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site((0, 0, 0), confidence=0.9),
            _hew_site((1, 0, 0), confidence=0.8),
            _hew_site((2, 0, 0), confidence=0.7),
        ])
        config = V6DConfig(top_k=2, min_confidence=0.3)
        guide = AnalyticESFieldGuide(site_map, config=config)

        self.assertEqual(guide._n_hew, 2,
                         f"Top-2 should keep 2 HEW sites, got {guide._n_hew}")

    def test_zero_hew_handling(self):
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        # No HEW sites
        site_map = _make_site_map([
            {"site_type": "stable_water", "center": [0, 0, 0],
             "radius": 1.4, "confidence": 0.8, "features": {}},
        ])
        config = V6DConfig()
        guide = AnalyticESFieldGuide(site_map, config=config)
        guide.to(self.torch.device("cpu"))

        self.assertEqual(guide._n_hew, 0)

        x = self.torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        h = self.torch.zeros(1, 11)
        h[0, 1] = 5.0

        energy = guide(self.torch.tensor(0.5), x=x, h=h, batch_mask=None)
        self.assertEqual(energy.item(), 0.0,
                         "Zero HEW should return zero energy")

    def test_mixed_filter(self):
        from models.analytic_esfield import AnalyticESFieldGuide, V6DConfig

        site_map = _make_site_map([
            _hew_site((0, 0, 0), env="mixed", confidence=0.5,
                      hbond_count=1, hydrophobic_contact_count=3,
                      nearest_protein_distance=4.0),
            _hew_site((1, 0, 0), env="hydrophobic", confidence=0.9),
        ])
        config = V6DConfig(filter_mixed=True, mixed_confidence_threshold=0.7, min_confidence=0.3)
        guide = AnalyticESFieldGuide(site_map, config=config)

        self.assertEqual(guide._n_hew, 1,
                         f"Low-confidence mixed HEW should be filtered, got {guide._n_hew}")


class TestV6DCompatibilityMatrix(unittest.TestCase):
    """Test that the compatibility matrix has correct structure."""

    def test_matrix_shape(self):
        from models.analytic_esfield import COMPAT_MATRIX, N_ATOM_TYPES, HEW_ENV_ORDER
        self.assertEqual(COMPAT_MATRIX.shape, (len(HEW_ENV_ORDER), N_ATOM_TYPES))

    def test_hydrophobic_rewards_carbon(self):
        from models.analytic_esfield import COMPAT_MATRIX, HEW_ENV_ORDER, ATOM_TYPE_TO_IDX
        env_idx = HEW_ENV_ORDER.index("hydrophobic")
        c_sp3 = ATOM_TYPE_TO_IDX["C_sp3"]
        c_arom = ATOM_TYPE_TO_IDX["C_aromatic"]
        self.assertGreater(COMPAT_MATRIX[env_idx, c_sp3].item(), 0.5)
        self.assertGreater(COMPAT_MATRIX[env_idx, c_arom].item(), 0.5)

    def test_hydrophobic_penalizes_charged(self):
        from models.analytic_esfield import COMPAT_MATRIX, HEW_ENV_ORDER, ATOM_TYPE_TO_IDX
        env_idx = HEW_ENV_ORDER.index("hydrophobic")
        charged = ATOM_TYPE_TO_IDX["charged"]
        self.assertLess(COMPAT_MATRIX[env_idx, charged].item(), 0.0)

    def test_polar_rewards_hbond_types(self):
        from models.analytic_esfield import COMPAT_MATRIX, HEW_ENV_ORDER, ATOM_TYPE_TO_IDX
        env_idx = HEW_ENV_ORDER.index("polar_unsatisfied")
        for at in ["O_acceptor", "N_donor", "N_acceptor"]:
            self.assertGreater(
                COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX[at]].item(), 0.5,
                f"polar_unsatisfied should reward {at}"
            )

    def test_buried_penalizes_bulky(self):
        from models.analytic_esfield import COMPAT_MATRIX, HEW_ENV_ORDER, ATOM_TYPE_TO_IDX
        env_idx = HEW_ENV_ORDER.index("buried")
        self.assertLess(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["C_aromatic"]].item(), 0.0)
        self.assertLess(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["charged"]].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
