"""Unit tests for src/guidance/two_stage_generation.py — v7 two-stage generation.

Tests cover:
  - TwoStageConfig / Phase1Config / Phase2Config dataclasses
  - AnchorAtoms representation
  - _compute_diagnostics: Phase 1 success check
  - _extract_anchors: anchor selection strategies
  - TwoStageGuideFn: Phase 2 composite guide function
  - _Phase1GuideFn: Phase 1 guidance
  - _tensors_from_rdmol: RDKit → tensor conversion
  - Integration: full two-stage pipeline (with mock model)
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guidance.latent_guidance import (
    SiteCompatibilityEnergy,
    build_site_energy_from_map,
    classify_hew_environment,
    hew_env_to_idx,
    ATOM_TYPE_VOCAB,
    ATOM_TYPE_TO_IDX,
    HEW_ENV_HYDROPHOBIC,
    HEW_ENV_POLAR_UNSATISFIED,
    HEW_ENV_MIXED,
)
from guidance.kinetic_trajectory_shaping import KTSScheduler
from guidance.two_stage_generation import (
    TwoStageConfig,
    Phase1Config,
    Phase2Config,
    AnchorAtoms,
    TwoStageGuideFn,
    _Phase1GuideFn,
    _compute_diagnostics,
    _extract_anchors,
    _tensors_from_rdmol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_site_map(centers, env_names, confidences=None):
    """Build a minimal site_map dict for testing."""
    sites = []
    for i, (center, env) in enumerate(zip(centers, env_names)):
        features = {}
        if env == HEW_ENV_HYDROPHOBIC:
            features = {"hbond_count": 0, "hydrophobic_contact_count": 4,
                        "nearest_protein_distance": 4.0}
        elif env == HEW_ENV_POLAR_UNSATISFIED:
            features = {"hbond_count": 0, "hydrophobic_contact_count": 1,
                        "nearest_protein_distance": 4.0}
        elif env == HEW_ENV_MIXED:
            features = {"hbond_count": 1, "hydrophobic_contact_count": 2,
                        "nearest_protein_distance": 4.0}
        sites.append({
            "site_id": i,
            "site_type": "high_energy_water",
            "center": list(center),
            "radius": 1.4,
            "score": 0.5,
            "confidence": confidences[i] if confidences else 0.8,
            "source": "test",
            "features": features,
        })
    return {"sites": sites}


# ---------------------------------------------------------------------------
# Test: Configuration dataclasses
# ---------------------------------------------------------------------------


class TestConfigurations(unittest.TestCase):
    """Test config dataclass construction and defaults."""

    def test_default_phase1_config(self):
        cfg = Phase1Config()
        self.assertEqual(cfg.n_init_atoms, 4)
        self.assertEqual(cfg.success_distance, 2.5)
        self.assertEqual(cfg.lambda_early, 0.5)

    def test_default_phase2_config(self):
        cfg = Phase2Config()
        self.assertEqual(cfg.restraint_force, 10.0)
        self.assertEqual(cfg.lambda_late, 0.1)

    def test_two_stage_config(self):
        cfg = TwoStageConfig()
        self.assertIsInstance(cfg.phase1, Phase1Config)
        self.assertIsInstance(cfg.phase2, Phase2Config)

    def test_custom_config(self):
        cfg = TwoStageConfig(
            phase1=Phase1Config(n_init_atoms=5, lambda_early=0.8),
            phase2=Phase2Config(restraint_force=20.0),
        )
        self.assertEqual(cfg.phase1.n_init_atoms, 5)
        self.assertEqual(cfg.phase2.restraint_force, 20.0)


# ---------------------------------------------------------------------------
# Test: AnchorAtoms
# ---------------------------------------------------------------------------


class TestAnchorAtoms(unittest.TestCase):
    """Test anchor atom representation."""

    def test_empty_anchors(self):
        anchors = AnchorAtoms(
            positions=torch.zeros(0, 3),
            type_indices=torch.zeros(0, dtype=torch.long),
            type_probs=torch.zeros(0, len(ATOM_TYPE_VOCAB)),
            occupied_sites=[],
            compat_scores=[],
            distances=[],
        )
        self.assertEqual(anchors.n_anchors, 0)

    def test_single_anchor(self):
        anchors = AnchorAtoms(
            positions=torch.tensor([[1.0, 0.0, 0.0]]),
            type_indices=torch.tensor([ATOM_TYPE_TO_IDX["C_sp3"]], dtype=torch.long),
            type_probs=torch.ones(1, len(ATOM_TYPE_VOCAB)) / len(ATOM_TYPE_VOCAB),
            occupied_sites=[0],
            compat_scores=[0.9],
            distances=[0.5],
        )
        self.assertEqual(anchors.n_anchors, 1)
        self.assertEqual(anchors.occupied_sites, [0])

    def test_to_dict_serializable(self):
        anchors = AnchorAtoms(
            positions=torch.tensor([[1.0, 0.0, 0.0]]),
            type_indices=torch.tensor([1], dtype=torch.long),
            type_probs=torch.ones(1, len(ATOM_TYPE_VOCAB)) / len(ATOM_TYPE_VOCAB),
            occupied_sites=[0],
            compat_scores=[0.9],
            distances=[0.5],
        )
        d = anchors.to_dict()
        self.assertEqual(d["n_anchors"], 1)
        self.assertEqual(len(d["positions"][0]), 3)
        self.assertIsInstance(d["compat_scores"][0], float)


# ---------------------------------------------------------------------------
# Test: _compute_diagnostics
# ---------------------------------------------------------------------------


class TestComputeDiagnostics(unittest.TestCase):
    """Test Phase 1 success detection logic."""

    def setUp(self):
        self.energy_fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        env_indices = torch.tensor([hew_env_to_idx(HEW_ENV_HYDROPHOBIC)])
        self.energy_fn.register_sites(centers, env_indices)

    def test_success_when_compatible_atom_near(self):
        """C_sp3 at site center → success."""
        x = torch.tensor([[0.0, 0.0, 0.0]])  # at site center
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        diag = _compute_diagnostics(x, h, self.energy_fn, 2.5, 0.5)
        self.assertTrue(diag["success"])
        self.assertEqual(diag["n_occupied_sites"], 1)
        self.assertLess(diag["best_distance"], 0.1)

    def test_failure_when_atom_too_far(self):
        """Atom far from site → failure."""
        x = torch.tensor([[10.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        diag = _compute_diagnostics(x, h, self.energy_fn, 2.5, 0.5)
        self.assertFalse(diag["success"])

    def test_failure_when_incompatible(self):
        """O_acceptor at hydrophobic site → failure (low compat)."""
        x = torch.tensor([[0.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["O_acceptor"]] = 1.0

        diag = _compute_diagnostics(x, h, self.energy_fn, 2.5, 0.5)
        self.assertFalse(diag["success"],
                         "Incompatible atom at hydrophobic site should not count as success")

    def test_mixed_type_probs_success(self):
        """Atom with mixed probs (mostly compatible) near site → success."""
        x = torch.tensor([[0.5, 0.0, 0.0]])
        logits = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        logits[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 3.0  # mostly C_sp3
        logits[0, ATOM_TYPE_TO_IDX["O_acceptor"]] = 0.1
        h = logits.softmax(dim=-1)

        diag = _compute_diagnostics(x, h, self.energy_fn, 2.5, 0.5)
        self.assertTrue(diag["success"])

    def test_no_sites(self):
        empty_fn = SiteCompatibilityEnergy()
        x = torch.randn(5, 3)
        h = torch.randn(5, len(ATOM_TYPE_VOCAB))
        diag = _compute_diagnostics(x, h, empty_fn, 2.5, 0.5)
        self.assertFalse(diag["success"])
        self.assertEqual(diag["n_sites"], 0)


# ---------------------------------------------------------------------------
# Test: _extract_anchors
# ---------------------------------------------------------------------------


class TestExtractAnchors(unittest.TestCase):
    """Test anchor atom extraction from Phase 1 output."""

    def setUp(self):
        self.energy_fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        centers = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        env_indices = torch.tensor([
            hew_env_to_idx(HEW_ENV_HYDROPHOBIC),
            hew_env_to_idx(HEW_ENV_POLAR_UNSATISFIED),
        ])
        self.energy_fn.register_sites(centers, env_indices)
        self.config = Phase1Config(
            success_distance=2.5,
            min_compatibility=0.3,
            anchor_selection="best_per_site",
        )

    def test_extract_none_on_failure(self):
        diag = {"success": False, "occupied_site_indices": []}
        x = torch.randn(3, 3)
        h = torch.randn(3, len(ATOM_TYPE_VOCAB))
        anchors = _extract_anchors(x, h, self.energy_fn, diag, self.config)
        self.assertIsNone(anchors)

    def test_extract_best_per_site(self):
        """Two atoms near two sites: each site gets its best atom."""
        x = torch.tensor([
            [0.5, 0.0, 0.0],   # near site 0 (hydrophobic)
            [5.0, 0.5, 0.0],   # near site 1 (polar)
            [10.0, 0.0, 0.0],  # far from both
        ])
        h = torch.zeros(3, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0       # compatible with site 0
        h[1, ATOM_TYPE_TO_IDX["O_acceptor"]] = 1.0   # compatible with site 1
        h[2, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0        # far from both

        diag = _compute_diagnostics(x, h, self.energy_fn, 2.5, 0.3)
        anchors = _extract_anchors(x, h, self.energy_fn, diag, self.config)

        self.assertIsNotNone(anchors)
        # Site 0 occupied by atom 0, site 1 by atom 1
        self.assertGreaterEqual(anchors.n_anchors, 1)

    def test_extract_all_compatible(self):
        cfg = Phase1Config(
            success_distance=2.5,
            min_compatibility=0.3,
            anchor_selection="all_compatible",
        )
        x = torch.tensor([
            [0.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        h = torch.zeros(3, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0
        h[1, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0
        h[2, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        diag = _compute_diagnostics(x, h, self.energy_fn, 2.5, 0.3)
        anchors = _extract_anchors(x, h, self.energy_fn, diag, cfg)

        self.assertIsNotNone(anchors)
        # Both nearby C_sp3 should be selected
        self.assertEqual(anchors.n_anchors, 2)

    def test_extract_nearest_compatible(self):
        cfg = Phase1Config(
            success_distance=2.5,
            min_compatibility=0.3,
            anchor_selection="nearest_compatible",
        )
        x = torch.tensor([
            [0.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ])
        h = torch.zeros(2, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0
        h[1, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        diag = _compute_diagnostics(x, h, self.energy_fn, 2.5, 0.3)
        anchors = _extract_anchors(x, h, self.energy_fn, diag, cfg)

        self.assertIsNotNone(anchors)
        self.assertEqual(anchors.n_anchors, 1, "nearest_compatible should return exactly 1")


# ---------------------------------------------------------------------------
# Test: _Phase1GuideFn
# ---------------------------------------------------------------------------


class TestPhase1GuideFn(unittest.TestCase):
    """Test the Phase 1 guide function callable."""

    def setUp(self):
        self.energy_fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        env_indices = torch.tensor([hew_env_to_idx(HEW_ENV_HYDROPHOBIC)])
        self.energy_fn.register_sites(centers, env_indices)

    def test_returns_scalar(self):
        guide = _Phase1GuideFn(self.energy_fn, lambda_guide=0.5,
                               guidance_start=0.0, guidance_end=1.0)
        x = torch.randn(3, 3)
        h = torch.randn(3, len(ATOM_TYPE_VOCAB))
        t = torch.tensor([0.5])

        result = guide(t, x=x, h=h, batch_mask=None)
        self.assertEqual(result.ndim, 0)
        self.assertTrue(torch.isfinite(result))

    def test_zero_outside_guidance_window(self):
        guide = _Phase1GuideFn(self.energy_fn, lambda_guide=0.5,
                               guidance_start=0.3, guidance_end=0.7)
        x = torch.randn(3, 3)
        h = torch.randn(3, len(ATOM_TYPE_VOCAB))

        r_early = guide(torch.tensor([0.1]), x=x, h=h, batch_mask=None)
        r_late = guide(torch.tensor([0.9]), x=x, h=h, batch_mask=None)

        self.assertEqual(r_early.item(), 0.0)
        self.assertEqual(r_late.item(), 0.0)

    def test_energy_sign_convention(self):
        """Guide returns -E_site so minimizing E = maximizing log_prob."""
        guide = _Phase1GuideFn(self.energy_fn, lambda_guide=1.0,
                               guidance_start=0.0, guidance_end=1.0)
        # C_sp3 at hydrophobic site center
        x = torch.tensor([[0.0, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        result = guide(torch.tensor([0.5]), x=x, h=h, batch_mask=None)

        # E_site ≈ -1.0 (compatible atom at center),
        # guide returns -E_site ≈ +1.0
        self.assertGreater(result.item(), 0.5)


# ---------------------------------------------------------------------------
# Test: TwoStageGuideFn (Phase 2 composite guide)
# ---------------------------------------------------------------------------


class TestTwoStageGuideFn(unittest.TestCase):
    """Test the Phase 2 composite guidance function."""

    def setUp(self):
        self.energy_fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        centers = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        env_indices = torch.tensor([
            hew_env_to_idx(HEW_ENV_HYDROPHOBIC),
            hew_env_to_idx(HEW_ENV_POLAR_UNSATISFIED),
        ])
        self.energy_fn.register_sites(centers, env_indices)

        self.anchors = AnchorAtoms(
            positions=torch.tensor([[0.0, 0.0, 0.0]]),
            type_indices=torch.tensor([ATOM_TYPE_TO_IDX["C_sp3"]], dtype=torch.long),
            type_probs=torch.ones(1, len(ATOM_TYPE_VOCAB)) / len(ATOM_TYPE_VOCAB),
            occupied_sites=[0],
            compat_scores=[0.9],
            distances=[0.1],
        )

        self.config = Phase2Config(
            lambda_late=0.1,
            restraint_force=10.0,
            guidance_start=0.1,
            guidance_end=0.9,
        )

    def test_returns_scalar(self):
        guide = TwoStageGuideFn(self.energy_fn, self.anchors, self.config)
        guide.set_anchor_indices([0], total_atoms=3)

        x = torch.randn(3, 3)
        h = torch.randn(3, len(ATOM_TYPE_VOCAB))

        result = guide(torch.tensor([0.5]), x=x, h=h, batch_mask=None)
        self.assertEqual(result.ndim, 0)
        self.assertTrue(torch.isfinite(result))

    def test_zero_outside_window(self):
        guide = TwoStageGuideFn(self.energy_fn, self.anchors, self.config)
        guide.set_anchor_indices([0], total_atoms=3)

        x = torch.randn(3, 3)
        h = torch.randn(3, len(ATOM_TYPE_VOCAB))

        r = guide(torch.tensor([0.05]), x=x, h=h, batch_mask=None)
        self.assertEqual(r.item(), 0.0)

    def test_restraint_activates(self):
        """Energy should include restraint when anchor deviates from target."""
        guide = TwoStageGuideFn(self.energy_fn, self.anchors, self.config)
        guide.set_anchor_indices([0], total_atoms=3)

        # Anchor at [1.0, 0, 0], should be at [0, 0, 0]
        x = torch.tensor([
            [1.0, 0.0, 0.0],  # anchor (deviated)
            [5.0, 0.0, 0.0],  # free atom
            [10.0, 0.0, 0.0], # free atom
        ])
        logits = torch.zeros(3, len(ATOM_TYPE_VOCAB))
        logits[:, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        r_with_restraint = guide(torch.tensor([0.5]), x=x, h=logits, batch_mask=None)

        # Without deviation: anchor at [0, 0, 0]
        x_ok = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        r_without = guide(torch.tensor([0.5]), x=x_ok, h=logits, batch_mask=None)

        # With deviation: more positive guide value (less negative energy)
        self.assertNotEqual(r_with_restraint.item(), r_without.item())

    def test_zero_restraint_when_disabled(self):
        cfg = Phase2Config(
            lambda_late=0.1,
            restraint_force=0.0,  # disabled
            guidance_start=0.1,
            guidance_end=0.9,
        )
        cfg_enabled = Phase2Config(
            lambda_late=0.1,
            restraint_force=10.0,  # enabled
            guidance_start=0.1,
            guidance_end=0.9,
        )
        guide_disabled = TwoStageGuideFn(self.energy_fn, self.anchors, cfg)
        guide_disabled.set_anchor_indices([0], total_atoms=3)
        guide_enabled = TwoStageGuideFn(self.energy_fn, self.anchors, cfg_enabled)
        guide_enabled.set_anchor_indices([0], total_atoms=3)

        # Same coordinates: anchor deviated from target [0,0,0] to [1,0,0]
        x = torch.tensor([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        h = torch.zeros(3, len(ATOM_TYPE_VOCAB))
        h[:, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        r_disabled = guide_disabled(torch.tensor([0.5]), x=x, h=h, batch_mask=None)
        r_enabled = guide_enabled(torch.tensor([0.5]), x=x, h=h, batch_mask=None)

        # With restraint enabled, the guide value should be different
        # (restraint punishes deviation → more positive energy → less positive guide return)
        self.assertNotEqual(r_disabled.item(), r_enabled.item(),
                            "Enabling restraint should change guide value when anchor deviates")

    def test_partial_anchor_indices(self):
        """Anchors can be a subset of all atoms (e.g., indices 1 and 3 of 5)."""
        guide = TwoStageGuideFn(self.energy_fn, self.anchors, self.config)
        guide.set_anchor_indices([1], total_atoms=5)

        x = torch.randn(5, 3)
        h = torch.randn(5, len(ATOM_TYPE_VOCAB))

        result = guide(torch.tensor([0.5]), x=x, h=h, batch_mask=None)
        self.assertTrue(torch.isfinite(result))


# ---------------------------------------------------------------------------
# Test: _tensors_from_rdmol
# ---------------------------------------------------------------------------


class TestTensorsFromRDMol(unittest.TestCase):
    """Test RDKit mol → tensor conversion."""

    def _make_mol(self, atoms):
        """Create an RDKit mol with 3D conformer from atomic numbers."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.RWMol()
        for i, atomic_num in enumerate(atoms):
            a = Chem.Atom(atomic_num)
            idx = mol.AddAtom(a)
        # Sanitize to compute implicit valences before AddHs
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)
        return mol

    def test_basic_conversion(self):
        mol = self._make_mol([6, 8])  # C, O
        x, h = _tensors_from_rdmol(mol)

        self.assertIsNotNone(x)
        self.assertIsNotNone(h)
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(h.shape[0], 2)
        self.assertEqual(h.shape[1], len(ATOM_TYPE_VOCAB))

    def test_hydrogen_exclusion(self):
        mol = self._make_mol([6])  # C only (hydrogens added then removed)
        x, h = _tensors_from_rdmol(mol)
        self.assertEqual(x.shape[0], 1, "Only heavy atom (C) should remain after RemoveHs")

    def test_type_mapping(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Use benzene for aromatic carbon (single atom can't be aromatic)
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)

        _, h = _tensors_from_rdmol(mol)
        # All 6 carbons should be C_aromatic (index 2)
        for i in range(6):
            self.assertGreater(h[i, ATOM_TYPE_TO_IDX["C_aromatic"]].item(), 0.5)


# ---------------------------------------------------------------------------
# Test: KTS scheduler integration
# ---------------------------------------------------------------------------


class TestKTSIntegration(unittest.TestCase):
    """Test KTS integration with guidance functions."""

    def test_kts_modulates_phase1_guidance(self):
        energy_fn = SiteCompatibilityEnergy(sigma_distance=3.0)
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        env_indices = torch.tensor([hew_env_to_idx(HEW_ENV_HYDROPHOBIC)])
        energy_fn.register_sites(centers, env_indices)

        # Default KTS: boost at t=0, damp at t=1
        guide_default = _Phase1GuideFn(energy_fn, lambda_guide=0.5,
                                       guidance_start=0.0, guidance_end=1.0,
                                       kts=KTSScheduler())

        # No KTS: eta=1 always
        guide_no_kts = _Phase1GuideFn(energy_fn, lambda_guide=0.5,
                                      guidance_start=0.0, guidance_end=1.0,
                                      kts=KTSScheduler(alpha0=0, beta0=0))

        x = torch.tensor([[0.5, 0.0, 0.0]])
        h = torch.zeros(1, len(ATOM_TYPE_VOCAB))
        h[0, ATOM_TYPE_TO_IDX["C_sp3"]] = 1.0

        # At t=0 (early), KTS should boost
        r_kts_early = guide_default(torch.tensor([0.0]), x=x, h=h, batch_mask=None)
        r_no_kts_early = guide_no_kts(torch.tensor([0.0]), x=x, h=h, batch_mask=None)
        self.assertNotEqual(r_kts_early.item(), r_no_kts_early.item(),
                            "KTS should modify guidance at t=0")

        # At t=0.6 (transition), both should be similar (eta≈1)
        r_kts_mid = guide_default(torch.tensor([0.6]), x=x, h=h, batch_mask=None)
        r_no_kts_mid = guide_no_kts(torch.tensor([0.6]), x=x, h=h, batch_mask=None)
        self.assertAlmostEqual(r_kts_mid.item(), r_no_kts_mid.item(), places=4,
                               msg="KTS should be near 1.0 at transition point")


if __name__ == "__main__":
    unittest.main()
