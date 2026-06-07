from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.build_atom_site_pairs import PairBuildConfig, build_atom_site_pairs  # noqa: E402
from evaluation.site_matching import evaluate_site_matching  # noqa: E402
from site_detection.build_crystal_water_sites import (  # noqa: E402
    CrystalWaterConfig,
    build_crystal_water_site_map,
)
from site_detection.merge_sites import merge_site_maps  # noqa: E402
from site_detection.site_schema import Site, SiteMap  # noqa: E402
from utils.structure_io import read_ligand_atoms, read_pdb_atoms  # noqa: E402


def pdb_atom(record: str, serial: int, name: str, res: str, chain: str, resid: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"{record:<6}{serial:5d} {name:^4} {res:>3} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2}\n"
    )


def toy_sdf() -> str:
    return """toy
  ESField

  3  0  0  0  0  0            999 V2000
    1.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""


class DataPipelineTest(unittest.TestCase):
    def test_crystal_water_pair_and_metric_pipeline(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            protein = tmp / "toy_rec.pdb"
            ligand = tmp / "toy_lig.sdf"
            protein.write_text(
                "".join(
                    [
                        pdb_atom("ATOM", 1, "N", "ALA", "A", 1, 0.0, 2.8, 0.0, "N"),
                        pdb_atom("ATOM", 2, "O", "ALA", "A", 1, 0.0, -2.8, 0.0, "O"),
                        pdb_atom("ATOM", 3, "C", "VAL", "A", 2, -2.0, 2.9, 0.0, "C"),
                        pdb_atom("ATOM", 4, "C", "VAL", "A", 2, -2.7, 0.0, 0.0, "C"),
                        pdb_atom("HETATM", 5, "O", "HOH", "A", 10, 1.0, 0.0, 0.0, "O"),
                        pdb_atom("HETATM", 6, "O", "HOH", "A", 11, -2.0, 0.0, 0.0, "O"),
                    ]
                ),
                encoding="utf-8",
            )
            ligand.write_text(toy_sdf(), encoding="utf-8")

            atoms = read_pdb_atoms(protein)
            ligand_atoms = read_ligand_atoms(ligand)
            site_map = build_crystal_water_site_map(
                protein,
                ligand_path=ligand,
                protein_id="toy_protein",
                ligand_id="toy_ligand",
                config=CrystalWaterConfig(max_sites=10, protein_clash_distance=0.5),
            )
            hydrophobic_map = SiteMap(
                protein_id="toy_protein",
                ligand_id="toy_ligand",
                pocket_center=(0.0, 0.0, 0.0),
                coordinate_frame="original_pdb_coordinates",
                sites=(
                    Site(
                        site_id=0,
                        site_type="hydrophobic_cavity",
                        center=(-2.6, 0.0, 0.0),
                        radius=1.5,
                        score=2.0,
                        confidence=0.9,
                        source="toy",
                        features={},
                    ),
                ),
            )
            merged = merge_site_maps([site_map, hydrophobic_map], merge_distance=0.4, max_sites=10)
            pairs = build_atom_site_pairs(
                ligand_atoms,
                merged,
                config=PairBuildConfig(negative_ratio=2, random_seed=7, split="train"),
            )
            metrics = evaluate_site_matching(ligand_atoms, merged)

        self.assertGreaterEqual(len(atoms), 6)
        self.assertGreaterEqual(len(site_map.sites), 1)
        self.assertGreater(sum(pair.label for pair in pairs), 0)
        self.assertGreater(len([pair for pair in pairs if pair.label == 0]), 0)
        self.assertGreater(metrics.site_occupancy_rate, 0.0)
        self.assertGreater(metrics.correct_atom_site_matching_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
