"""Unit tests for src/guidance/latent_guidance.py — v7 site-compatibility guidance.

Tests cover:
  - SiteCompatibilityEnergy: energy computation, gradients, per-site scores
  - apply_latent_guidance: velocity correction with KTS
  - TypeGuidanceBias: logit biasing
  - Compatibility matrix: correctness of hard-coded values
  - build_site_energy_from_map: site map → energy conversion
  - harmonic_restraint_energy: anchor restraint energy
"""

import math
import unittest

import torch

# Ensure src is on path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guidance.latent_guidance import (
    SiteCompatibilityEnergy,
    TypeGuidanceBias,
    apply_latent_guidance,
    build_site_energy_from_map,
    harmonic_restraint_energy,
    classify_hew_environment,
    hew_env_to_idx,
    COMPAT_MATRIX,
    ATOM_TYPE_VOCAB,
    ATOM_TYPE_TO_IDX,
    HEW_ENV_HYDROPHOBIC,
    HEW_ENV_POLAR_UNSATISFIED,
    HEW_ENV_MIXED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_site_map(centers, env_names, confidences=None):
    """Build a minimal site_map dict for testing."""
    sites = []
    for i, (center, env) in enumerate(zip(centers, env_names)):
        sites.append({
            "site_id": i,
            "site_type": "high_energy_water",
            "center": list(center),
            "radius": 1.4,
            "score": 0.5,
            "confidence": confidences[i] if confidences else 0.8,
            "source": "test",
            "features": _env_features(env),
        })
    return {"sites": sites}


def _env_features(env):
    """Return features dict that classify_hew_environment maps to `env`."""
    if env == HEW_ENV_HYDROPHOBIC:
        return {"hbond_count": 0, "hydrophobic_contact_count": 4, "nearest_protein_distance": 4.0}
    elif env == HEW_ENV_POLAR_UNSATISFIED:
        return {"hbond_count": 0, "hydrophobic_contact_count": 1, "nearest_protein_distance": 4.0}
    elif env == HEW_ENV_MIXED:
        return {"hbond_count": 1, "hydrophobic_contact_count": 2, "nearest_protein_distance": 4.0}
    else:
        return {"hbond_count": 0, "hydrophobic_contact_count": 0, "nearest_protein_distance": 2.0}


# ---------------------------------------------------------------------------
# Test: Compatibility matrix
# ---------------------------------------------------------------------------


class TestCompatibilityMatrix(unittest.TestCase):
    """Verify the hard-coded compatibility matrix has correct structure."""

    def test_matrix_shape(self):
        self.assertEqual(COMPAT_MATRIX.shape, (4, len(ATOM_TYPE_VOCAB)))

    def test_hydrophobic_compat_values(self):
        """Hydrophobic sites should strongly prefer C and halogens, penalize polar."""
        env_idx = hew_env_to_idx(HEW_ENV_HYDROPHOBIC)
        # Compatible: C_sp3, C_aromatic, halogen
        self.assertGreater(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["C_sp3"]], 0.5)
        self.assertGreater(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["C_aromatic"]], 0.5)
        self.assertGreater(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["halogen"]], 0.5)
        # Incompatible: O_acceptor, N_donor
        self.assertLess(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["O_acceptor"]], 0.0)
        self.assertLess(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["N_donor"]], 0.0)
        # Strongly incompatible: charged
        self.assertLess(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["charged"]], -0.5)

    def test_polar_unsatisfied_compat_values(self):
        """Polar-unsatisfied sites should prefer H-bond donors/acceptors."""
        env_idx = hew_env_to_idx(HEW_ENV_POLAR_UNSATISFIED)
        self.assertGreater(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["O_acceptor"]], 0.5)
        self.assertGreater(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["N_donor"]], 0.5)
        self.assertLess(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["C_sp3"]], 0.0)
        self.assertLess(COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX["charged"]], -0.5)

    def test_matrix_symmetric_patterns(self):
        """Hydrophobic-positive types should be polar-negative and vice versa."""
        h_idx = hew_env_to_idx(HEW_ENV_HYDROPHOBIC)
        p_idx = hew_env_to_idx(HEW_ENV_POLAR_UNSATISFIED)
        # C_sp3: positive for hydrophobic, negative for polar
        self.assertGreater(COMPAT_MATRIX[h_idx, ATOM_TYPE_TO_IDX["C_sp3"]], 0)
        self.assertLess(COMPAT_MATRIX[p_idx, ATOM_TYPE_TO_IDX["C_sp3"]], 0)
        # O_acceptor: negative for hydrophobic, positive for polar
        self.assertLess(COMPAT_MATRIX[h_idx, ATOM_TYPE_TO_IDX["O_acceptor"]], 0)
        self.assertGreater(COMPAT_MATRIX[p_idx, ATOM_TYPE_TO_IDX["O_acceptor"]], 0)

    def test_mixed_env_all_moderate(self):
        """Mixed environment should have moderate scores for most types."""
        env_idx = hew_env_to_idx(HEW_ENV_MIXED)
        # All common types should have non-negative scores
        for atype in ["C_sp3", "C_aromatic", "O_acceptor", "N_donor", "halogen", "S"]:
            self.assertGreaterEqual(
                COMPAT_MATRIX[env_idx, ATOM_TYPE_TO_IDX[atype]], 0.0,
                f"Mixed env should have non-negative compat for {atype}"
            )


# ---------------------------------------------------------------------------
# Test: classify_hew_environment
# ---------------------------------------------------------------------------


class TestEnvironmentClassification(unittest.TestCase):
    """Test HEW environment classification logic."""

    def test_hydrophobic_classification(self):
        site = {"features": {"hbond_count": 1, "hydrophobic_contact_count": 4, "nearest_protein_distance": 4.0}}
        self.assertEqual(classify_hew_environment(site), HEW_ENV_HYDROPHOBIC)

    def test_polar_unsatisfied_classification(self):
        site = {"features": {"hbond_count": 1, "hydrophobic_contact_count": 2, "nearest_protein_distance": 4.0}}
        self.assertEqual(classify_hew_environment(site), HEW_ENV_POLAR_UNSATISFIED)

    def test_mixed_classification(self):
        site = {"features": {"hbond_count": 2, "hydrophobic_contact_count": 4, "nearest_protein_distance": 4.0}}
        self.assertEqual(classify_hew_environment(site), HEW_ENV_MIXED)

    def test_buried_classification(self):
        site = {"features": {"hbond_count": 0, "hydrophobic_contact_count": 0, "nearest_protein_distance": 2.0}}
        self.assertEqual(classify_hew_environment(site), "buried")


# ---------------------------------------------------------------------------
# Test: SiteCompatibilityEnergy
# ---------------------------------------------------------------------------


class TestSiteCompatibilityEnergy(unittest.TestCase):
    """Test the differentiable site-compatibility energy function."""

    def setUp(self):
        self.sigma = 3.0
        self.energy_fn = SiteCompatibilityEnergy(sigma_distance=self.sigma)
        # One hydrophobic site at origin
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        env_indices = torch.tensor([hew_env_to_idx(HEW_ENV_HYDROPHOBIC)])
        self.energy_fn.register_sites(centers, env_indices)

    def test_energy_zero_when_no_sites(self):
        empty_fn = SiteCompatibilityEnergy()
        x = torch.randn(5, 3)
        h = torch.randn(5, len(ATOM_TYPE_VOCAB))
        e = empty_fn(x, atom_type_probs=h.softmax(dim=-1))
        self.assertEqual(e.item(), 0.0)

    def test_compatible_atom_lowers_energy(self):
        """A compatible atom (C_sp3) near a hydrophobic site should lower energy."""
        # Place C_sp3 atom at the site center
        x = torch.tensor([[0.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        e = self.energy_fn(x, atom_type_probs=h)
        # Energy = -compat * exp(-0/(2σ²)) = -1.0 * 1.0 = -1.0
        self.assertLess(e.item(), -0.5, "Compatible atom at site center should have negative energy")

    def test_incompatible_atom_raises_energy(self):
        """An incompatible atom (O_acceptor) near hydrophobic site should raise energy."""
        x = torch.tensor([[0.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["O_acceptor"]] = 1.0

        e = self.energy_fn(x, atom_type_probs=h)
        # Energy = -(-0.5) * 1.0 = 0.5 (positive = unfavorable)
        self.assertGreater(e.item(), 0.0, "Incompatible atom at site center should have positive energy")

    def test_energy_decays_with_distance(self):
        """Energy magnitude should decay as atom moves away from site."""
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        e_near = self.energy_fn(torch.tensor([[0.0, 0.0, 0.0]]), atom_type_probs=h)
        e_far = self.energy_fn(torch.tensor([[10.0, 0.0, 0.0]]), atom_type_probs=h)

        # Near: E ≈ -1.0, Far: E ≈ 0 (gaussian decay)
        self.assertLess(e_near.item(), e_far.item(),
                        "Energy should be lower (more negative) when atom is closer")
        # At d=10, sigma=3: exp(-100/18) ≈ 0.004, so energy ≈ -0.004
        self.assertGreater(e_far.item(), -0.01,
                           "Energy should be near zero at large distance")

    def test_gradient_direction(self):
        """∇E points AWAY from site for compatible atoms (energy decreases near site).

        The raw gradient ∇_x E points in the direction of increasing energy.
        For compatible atoms, energy DECREASES near the site (more negative),
        so ∇E points AWAY from the site center.  The guidance step uses
        -λ∇E, which then pulls atoms TOWARD the site.
        """
        x = torch.tensor([[2.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        energy, grad = self.energy_fn.compute_gradient(x, atom_type_probs=h, grad_clip=10.0)

        # Site at origin, atom at x=2.0.  Compat=1.0 → ∇E > 0 (energy higher
        # as atom moves right, away from site).  Guidance: -λ∇E < 0 pulls left.
        self.assertGreater(grad[0, 0].item(), 0.0,
                           f"∇E should be positive x (away from site at origin), got {grad[0, 0].item():.4f}")

    def test_gradient_pushes_incompatible_away(self):
        """∇E points TOWARD site for incompatible atoms (energy higher near site).

        For incompatible atoms, energy INCREASES near the site, so ∇E points
        TOWARD the site center.  The guidance -λ∇E then pushes atoms AWAY.
        """
        x = torch.tensor([[1.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["O_acceptor"]] = 1.0

        energy, grad = self.energy_fn.compute_gradient(x, atom_type_probs=h, grad_clip=10.0)

        # Site at origin, atom at x=1.0.  Compat=−0.5 → ∇E < 0 (energy lower
        # as atom moves right, away from incompatible site).
        # Guidance: -λ∇E > 0 pushes right (away from site).
        self.assertLess(grad[0, 0].item(), 0.0,
                        f"∇E should be negative x (toward incompatible site), got {grad[0, 0].item():.4f}")

    def test_gradient_clipping(self):
        """Gradient clipping should limit per-atom gradient norm."""
        x = torch.tensor([[1.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        _, grad = self.energy_fn.compute_gradient(x, atom_type_probs=h, grad_clip=0.1)
        gnorm = grad.norm(dim=-1).max()
        self.assertLessEqual(gnorm.item(), 0.1 + 1e-5)

    def test_soft_type_probs(self):
        """Energy should work with soft type probabilities (not just one-hot)."""
        x = torch.tensor([[0.0, 0.0, 0.0]])
        # Logits: C_sp3 favored (2.0), O_acceptor mild (0.5), others 0
        logits = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        logits[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 2.0
        logits[0, ATOM_TYPE_TO_IDX["O_acceptor"]] = 0.5
        probs = logits.softmax(dim=-1)

        e = self.energy_fn(x, atom_type_probs=probs)

        # Expected: C_sp3 prob ~0.41, O_acceptor prob ~0.09, rest ~0.05 each.
        # Hydrophobic site: C_sp3(+1.0), C_aromatic(+1.0), halogen(+1.0),
        # O_acceptor(-0.5), N_donor(-0.5), etc.
        # Rough expected compat ~0.32, so E ≈ -0.32
        self.assertLess(e.item(), -0.25)
        self.assertGreater(e.item(), -0.9)

    def test_hard_type_indices(self):
        """Energy should work with hard type indices."""
        x = torch.tensor([[0.0, 0.0, 0.0]])
        type_idx = torch.tensor([ATOM_TYPE_TO_IDX["C_sp3"]], dtype=torch.long)

        e = self.energy_fn(x, atom_type_indices=type_idx)
        self.assertLess(e.item(), -0.5)

    def test_multiple_sites(self):
        """Energy should correctly aggregate over multiple sites."""
        centers = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        env_indices = torch.tensor([
            hew_env_to_idx(HEW_ENV_HYDROPHOBIC),
            hew_env_to_idx(HEW_ENV_POLAR_UNSATISFIED),
        ])

        fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        fn.register_sites(centers, env_indices)

        # C_sp3 near hydrophobic site, far from polar
        x = torch.tensor([[0.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        e = fn(x, atom_type_probs=h)
        # Site 0 (hydrophobic): compat=1.0, gauss(0)=1.0 → contribution -1.0
        # Site 1 (polar at d=5): compat≈-0.5, gauss(5)=exp(-25/18)≈0.25 → contribution +0.125
        # Total ≈ -0.875
        self.assertLess(e.item(), -0.7)
        self.assertGreater(e.item(), -1.0)

    def test_per_site_scores(self):
        """Per-site scores should reflect individual site occupancy."""
        centers = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        env_indices = torch.tensor([
            hew_env_to_idx(HEW_ENV_HYDROPHOBIC),
            hew_env_to_idx(HEW_ENV_POLAR_UNSATISFIED),
        ])

        fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        fn.register_sites(centers, env_indices)

        x = torch.tensor([[0.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        scores = fn.compute_per_site_scores(x, atom_type_probs=h)
        self.assertEqual(scores.shape[0], 2)
        # Site 0 (hydrophobic, compat=1, gauss=1): score ≈ 1.0
        self.assertGreater(scores[0].item(), 0.5)
        # Site 1 (polar at d=5, compat=-0.5, gauss≈0.25): score ≈ -0.125
        self.assertLess(scores[1].item(), 0.0)

    def test_differentiable(self):
        """Energy must be twice-differentiable (for gradient guidance)."""
        x = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        e = self.energy_fn(x, atom_type_probs=h)
        grad = torch.autograd.grad(e, x, create_graph=True)[0]
        # Second derivative should exist
        grad_norm = grad.norm()
        grad_norm.backward()
        self.assertIsNotNone(x.grad)


# ---------------------------------------------------------------------------
# Test: apply_latent_guidance
# ---------------------------------------------------------------------------


class TestApplyLatentGuidance(unittest.TestCase):
    """Test the Metadiffusion-style velocity correction."""

    def setUp(self):
        self.energy_fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        env_indices = torch.tensor([hew_env_to_idx(HEW_ENV_HYDROPHOBIC)])
        self.energy_fn.register_sites(centers, env_indices)

    def test_guidance_modifies_velocity(self):
        """Guided velocity should differ from base velocity when λ > 0."""
        x = torch.tensor([[1.0, 0.0, 0.0]])
        v = torch.tensor([[0.1, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        result = apply_latent_guidance(
            x, v, self.energy_fn, t=0.3,
            atom_type_probs=h, lambda_guide=0.5,
        )

        # Velocity should be modified
        diff = (result["guided_velocity"] - v).norm()
        self.assertGreater(diff.item(), 0.0)

    def test_zero_lambda_no_guidance(self):
        """λ=0 should return base velocity unchanged."""
        x = torch.tensor([[1.0, 0.0, 0.0]])
        v = torch.tensor([[0.1, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        result = apply_latent_guidance(
            x, v, self.energy_fn, t=0.3,
            atom_type_probs=h, lambda_guide=0.0,
        )

        self.assertTrue(torch.allclose(result["guided_velocity"], v))
        self.assertEqual(result["lambda_effective"], 0.0)

    def test_kts_scaling(self):
        """KTS factor η should modulate the effective guidance strength."""
        x = torch.tensor([[1.0, 0.0, 0.0]])
        v = torch.tensor([[0.1, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        r1 = apply_latent_guidance(x, v, self.energy_fn, t=0.3,
                                   atom_type_probs=h, lambda_guide=0.5, kts_eta=1.0)
        r2 = apply_latent_guidance(x, v, self.energy_fn, t=0.3,
                                   atom_type_probs=h, lambda_guide=0.5, kts_eta=2.0)

        # Higher eta = larger velocity modification
        diff1 = (r1["guided_velocity"] - v).norm()
        diff2 = (r2["guided_velocity"] - v).norm()
        self.assertGreater(diff2.item(), diff1.item(),
                           "Higher KTS eta should produce larger velocity change")

    def test_no_sites_no_guidance(self):
        """When no sites are registered, guidance should be a no-op."""
        empty_fn = SiteCompatibilityEnergy()
        x = torch.randn(5, 3)
        v = torch.randn(5, 3)

        result = apply_latent_guidance(x, v, empty_fn, t=0.5, lambda_guide=0.5)
        self.assertTrue(torch.allclose(result["guided_velocity"], v))


# ---------------------------------------------------------------------------
# Test: TypeGuidanceBias
# ---------------------------------------------------------------------------


class TestTypeGuidanceBias(unittest.TestCase):
    """Test the atom-type logit biasing mechanism."""

    def setUp(self):
        self.energy_fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        env_indices = torch.tensor([hew_env_to_idx(HEW_ENV_HYDROPHOBIC)])
        self.energy_fn.register_sites(centers, env_indices)
        self.type_guidance = TypeGuidanceBias(self.energy_fn, lambda_type=0.5)

    def test_bias_increases_compatible_logits(self):
        """Compatible type logits should increase when atom is near site."""
        x = torch.tensor([[0.5, 0.0, 0.0]])
        logits = torch.zeros(1, len(ATOM_TYPE_VOCAB))

        biased = self.type_guidance(logits, x)

        # C_sp3 should get positive bias (compatible with hydrophobic)
        c_idx = ATOM_TYPE_TO_IDX["C_sp3"]
        self.assertGreater(biased[0, c_idx].item(), logits[0, c_idx].item())

    def test_bias_decreases_incompatible_logits(self):
        """Incompatible type logits should decrease near the wrong site."""
        x = torch.tensor([[0.5, 0.0, 0.0]])
        logits = torch.zeros(1, len(ATOM_TYPE_VOCAB))

        biased = self.type_guidance(logits, x)

        # O_acceptor should get negative bias (incompatible with hydrophobic)
        o_idx = ATOM_TYPE_TO_IDX["O_acceptor"]
        self.assertLess(biased[0, o_idx].item(), logits[0, o_idx].item())

    def test_bias_decays_with_distance(self):
        """Type bias should decay when atom is far from any site."""
        x_near = torch.tensor([[0.5, 0.0, 0.0]])
        x_far = torch.tensor([[10.0, 0.0, 0.0]])
        logits = torch.zeros(1, len(ATOM_TYPE_VOCAB))

        b_near = self.type_guidance(logits.clone(), x_near)
        b_far = self.type_guidance(logits.clone(), x_far)

        c_idx = ATOM_TYPE_TO_IDX["C_sp3"]
        near_bias = b_near[0, c_idx] - logits[0, c_idx]
        far_bias = b_far[0, c_idx] - logits[0, c_idx]
        self.assertGreater(abs(near_bias.item()), abs(far_bias.item()),
                           "Bias should be stronger when atom is near the site")

    def test_zero_lambda_no_bias(self):
        """λ_type=0 should return logits unchanged."""
        tg = TypeGuidanceBias(self.energy_fn, lambda_type=0.0)
        x = torch.tensor([[0.5, 0.0, 0.0]])
        logits = torch.randn(1, len(ATOM_TYPE_VOCAB))
        biased = tg(logits, x)
        self.assertTrue(torch.allclose(biased, logits))


# ---------------------------------------------------------------------------
# Test: build_site_energy_from_map
# ---------------------------------------------------------------------------


class TestBuildSiteEnergyFromMap(unittest.TestCase):
    """Test conversion from site_map dict to SiteCompatibilityEnergy."""

    def test_empty_site_map(self):
        site_map = {"sites": []}
        energy = build_site_energy_from_map(site_map)
        self.assertEqual(energy.n_sites, 0)

    def test_no_hew_sites(self):
        site_map = {"sites": [
            {"site_id": 0, "site_type": "stable_water", "center": [0, 0, 0],
             "radius": 1.4, "score": 0.5, "confidence": 0.8, "source": "test",
             "features": {}}
        ]}
        energy = build_site_energy_from_map(site_map)
        self.assertEqual(energy.n_sites, 0, "Non-HEW sites should be filtered out")

    def test_hew_sites_registered(self):
        site_map = _make_site_map(
            [(0, 0, 0)], [HEW_ENV_HYDROPHOBIC]
        )
        energy = build_site_energy_from_map(site_map)
        self.assertEqual(energy.n_sites, 1)

    def test_confidence_filter(self):
        site_map = {
            "sites": [
                {"site_id": 0, "site_type": "high_energy_water",
                 "center": [0, 0, 0], "radius": 1.4, "score": 0.5,
                 "confidence": 0.2, "source": "test",
                 "features": _env_features(HEW_ENV_HYDROPHOBIC)},
                {"site_id": 1, "site_type": "high_energy_water",
                 "center": [5, 0, 0], "radius": 1.4, "score": 0.5,
                 "confidence": 0.9, "source": "test",
                 "features": _env_features(HEW_ENV_HYDROPHOBIC)},
            ]
        }
        energy = build_site_energy_from_map(site_map, min_confidence=0.5)
        self.assertEqual(energy.n_sites, 1, "Only high-confidence site should remain")

    def test_env_filter(self):
        site_map = {
            "sites": [
                {"site_id": 0, "site_type": "high_energy_water",
                 "center": [0, 0, 0], "radius": 1.4, "score": 0.5,
                 "confidence": 0.8, "source": "test",
                 "features": _env_features(HEW_ENV_HYDROPHOBIC)},
                {"site_id": 1, "site_type": "high_energy_water",
                 "center": [5, 0, 0], "radius": 1.4, "score": 0.5,
                 "confidence": 0.8, "source": "test",
                 "features": _env_features(HEW_ENV_POLAR_UNSATISFIED)},
            ]
        }
        energy = build_site_energy_from_map(
            site_map, enabled_envs=(HEW_ENV_HYDROPHOBIC,)
        )
        self.assertEqual(energy.n_sites, 1, "Only hydrophobic sites should remain")

    def test_top_k(self):
        site_map = {
            "sites": [
                {"site_id": i, "site_type": "high_energy_water",
                 "center": [float(i), 0, 0], "radius": 1.4, "score": 0.5,
                 "confidence": 0.5 + 0.1 * i, "source": "test",
                 "features": _env_features(HEW_ENV_HYDROPHOBIC)}
                for i in range(10)
            ]
        }
        energy = build_site_energy_from_map(site_map, top_k=3)
        self.assertEqual(energy.n_sites, 3)


# ---------------------------------------------------------------------------
# Test: harmonic_restraint_energy
# ---------------------------------------------------------------------------


class TestHarmonicRestraintEnergy(unittest.TestCase):
    """Test harmonic restraint for anchor atoms."""

    def test_zero_when_no_fixed_atoms(self):
        x = torch.randn(5, 3)
        mask = torch.zeros(5, dtype=torch.bool)
        target = torch.zeros(5, 3)
        e = harmonic_restraint_energy(x, mask, target)
        self.assertEqual(e.item(), 0.0)

    def test_energy_proportional_to_squared_distance(self):
        x = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mask = torch.tensor([True, False])
        target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        k = 10.0

        e = harmonic_restraint_energy(x, mask, target, force_constant=k)
        # Only first atom contributes: k * (1² + 0² + 0²) = 10.0
        self.assertAlmostEqual(e.item(), 10.0, places=5)

    def test_energy_increases_with_distance(self):
        x1 = torch.tensor([[1.0, 0.0, 0.0]])
        x2 = torch.tensor([[3.0, 0.0, 0.0]])
        mask = torch.tensor([True])
        target = torch.tensor([[0.0, 0.0, 0.0]])

        e1 = harmonic_restraint_energy(x1, mask, target, force_constant=1.0)
        e2 = harmonic_restraint_energy(x2, mask, target, force_constant=1.0)

        self.assertGreater(e2.item(), e1.item())

    def test_differentiable(self):
        x = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        mask = torch.tensor([True])
        target = torch.tensor([[0.0, 0.0, 0.0]])

        e = harmonic_restraint_energy(x, mask, target)
        grad = torch.autograd.grad(e, x)[0]
        # Force = -k * (x - x0) = -10 * (1, 0, 0) → grad = (20, 0, 0)
        # Actually E = k * sum((x-x0)^2), dE/dx = 2k*(x-x0)
        # k=10: dE/dx1 = 20
        self.assertAlmostEqual(grad[0, 0].item(), 20.0, places=5)
        self.assertAlmostEqual(grad[0, 1].item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
