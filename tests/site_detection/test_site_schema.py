import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from site_detection.site_schema import SCHEMA_VERSION, Site, SiteMap


class TestSiteSchema(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        site_map = SiteMap(
            protein_id="1abc_A",
            ligand_id="1abc_lig",
            pocket_center=[12.3, 4.5, -8.1],
            coordinate_frame="original_pdb_coordinates",
            sites=[
                Site(
                    site_id=0,
                    site_type="high_energy_water",
                    center=[11.9, 4.0, -7.5],
                    radius=1.4,
                    score=2.1,
                    confidence=0.78,
                    source="crystal_water_rule",
                    features={"occupancy": 0.35, "enthalpy": 1.7, "entropy": 0.4},
                ),
                Site(
                    site_id=1,
                    site_type="hydrophobic_cavity",
                    center=[10.2, 3.1, -6.9],
                    radius=1.8,
                    score=1.2,
                    confidence=0.62,
                    source="fpocket",
                    features={"hydrophobic_score": 0.91},
                ),
            ],
        )

        payload = site_map.to_json()
        loaded = SiteMap.from_json(payload)

        self.assertEqual(site_map, loaded)
        self.assertEqual(SCHEMA_VERSION, loaded.schema_version)


if __name__ == "__main__":
    unittest.main()
