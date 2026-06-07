from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from site_detection.site_schema import (  # noqa: E402
    SCHEMA_VERSION,
    SchemaValidationError,
    Site,
    SiteMap,
)


def toy_site_map() -> SiteMap:
    return SiteMap(
        schema_version=SCHEMA_VERSION,
        protein_id="toy_protein_A",
        ligand_id="toy_ligand",
        pocket_center=(0.0, 0.0, 0.0),
        coordinate_frame="original_pdb_coordinates",
        sites=(
            Site(
                site_id=0,
                site_type="high_energy_water",
                center=(1.0, 0.0, 0.0),
                radius=1.4,
                score=2.1,
                confidence=0.8,
                source="crystal_water_rule",
                features={"hbond_count": 1, "buriedness": 0.6},
            ),
            Site(
                site_id=1,
                site_type="hydrophobic_cavity",
                center=(-1.0, 0.2, 0.0),
                radius=1.8,
                score=1.3,
                confidence=0.7,
                source="fpocket",
                features={"alpha_sphere_count": 5},
            ),
        ),
    )


class SiteSchemaTest(unittest.TestCase):
    def test_json_string_round_trip(self) -> None:
        site_map = toy_site_map()

        restored = SiteMap.from_json(site_map.to_json())

        self.assertEqual(restored, site_map)
        payload = json.loads(restored.to_json())
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(payload["sites"]), 2)

    def test_json_file_round_trip(self) -> None:
        site_map = toy_site_map()
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "toy_site_map.json"

            site_map.write_json(path)
            restored = SiteMap.read_json(path)

        self.assertEqual(restored, site_map)

    def test_rejects_invalid_site_type(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "site_type"):
            Site(
                site_id=0,
                site_type="salt_bridge",  # type: ignore[arg-type]
                center=(0.0, 0.0, 0.0),
                radius=1.0,
                score=0.0,
                confidence=0.5,
                source="toy",
                features={},
            )

    def test_normalizes_direct_constructor_values(self) -> None:
        site = Site(
            site_id=0,
            site_type="stable_water",
            center=[0, 1, 2],  # type: ignore[arg-type]
            radius=1,
            score=-1,
            confidence=1,
            source="toy",
            features={},
        )
        site_map = SiteMap(
            protein_id="toy_protein_A",
            ligand_id="toy_ligand",
            pocket_center=[0, 0, 0],  # type: ignore[arg-type]
            coordinate_frame="original_pdb_coordinates",
            sites=[site],  # type: ignore[arg-type]
        )

        self.assertEqual(site.center, (0.0, 1.0, 2.0))
        self.assertEqual(site.radius, 1.0)
        self.assertEqual(site.score, -1.0)
        self.assertEqual(site.confidence, 1.0)
        self.assertEqual(site_map.pocket_center, (0.0, 0.0, 0.0))
        self.assertIsInstance(site_map.sites, tuple)

    def test_rejects_duplicate_site_ids(self) -> None:
        first = toy_site_map().sites[0]
        duplicate = Site(
            site_id=first.site_id,
            site_type="stable_water",
            center=(0.0, 1.0, 0.0),
            radius=1.2,
            score=-1.0,
            confidence=0.9,
            source="crystal_water_rule",
            features={},
        )

        with self.assertRaisesRegex(SchemaValidationError, "unique site_id"):
            SiteMap(
                protein_id="toy_protein_A",
                ligand_id="toy_ligand",
                pocket_center=(0.0, 0.0, 0.0),
                coordinate_frame="original_pdb_coordinates",
                sites=(first, duplicate),
            )

    def test_rejects_missing_required_key(self) -> None:
        payload = toy_site_map().to_dict()
        del payload["sites"][0]["features"]

        with self.assertRaisesRegex(SchemaValidationError, "missing required keys"):
            SiteMap.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
