from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ModelGuidanceTest(unittest.TestCase):
    def test_potential_forward_backward_and_guidance(self) -> None:
        try:
            import torch
            from guidance.energy_guidance import EnergyGuidance, EnergyGuidanceConfig
            from models.atom_features import atom_type_to_index
            from models.potential_network import CompatibilityPotential, PotentialConfig
            from site_detection.site_schema import Site, SiteMap
        except ImportError as exc:
            self.skipTest(f"PyTorch not installed: {exc}")

        model = CompatibilityPotential(PotentialConfig(hidden_dim=32, num_layers=1, rbf_bins=4))
        coordinates = torch.tensor([[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
        atom_type_indices = torch.tensor([atom_type_to_index("O_acceptor"), atom_type_to_index("C_sp3")])
        site_map = SiteMap(
            protein_id="toy_protein",
            ligand_id="toy_ligand",
            pocket_center=(0.0, 0.0, 0.0),
            coordinate_frame="original_pdb_coordinates",
            sites=(
                Site(
                    site_id=0,
                    site_type="high_energy_water",
                    center=(0.0, 0.0, 0.0),
                    radius=1.4,
                    score=1.0,
                    confidence=0.8,
                    source="toy",
                    features={},
                ),
            ),
        )

        guidance = EnergyGuidance(model, EnergyGuidanceConfig(grad_clip=0.5))
        energy, grad = guidance.coordinate_gradient(coordinates, site_map=site_map, atom_type_indices=atom_type_indices)
        self.assertTrue(torch.isfinite(energy))
        self.assertEqual(tuple(grad.shape), tuple(coordinates.shape))
        self.assertLessEqual(float(grad.norm(dim=-1).max()), 0.5001)


if __name__ == "__main__":
    unittest.main()
