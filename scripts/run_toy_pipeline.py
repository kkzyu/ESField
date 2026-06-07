"""Run a dependency-light ESField toy pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.atom_site_schema import write_pairs_jsonl  # noqa: E402
from data.build_atom_site_pairs import PairBuildConfig, build_atom_site_pairs  # noqa: E402
from evaluation.failure_analysis import analyze_site_metric_rows  # noqa: E402
from evaluation.site_matching import evaluate_site_matching, write_metrics_outputs  # noqa: E402
from site_detection.build_crystal_water_sites import CrystalWaterConfig, build_crystal_water_site_map  # noqa: E402
from site_detection.merge_sites import merge_site_maps  # noqa: E402
from site_detection.site_schema import Site, SiteMap  # noqa: E402
from utils.structure_io import read_ligand_atoms  # noqa: E402
from visualization.export_pymol import write_pymol_script  # noqa: E402
from visualization.plot_site_map import plot_site_map  # noqa: E402
from visualization.report import generate_html_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESField toy pipeline.")
    parser.add_argument("--output-dir", default="experiments/toy_pipeline")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output dir is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    protein = output_dir / "toy_rec.pdb"
    ligand = output_dir / "toy_lig.sdf"
    protein.write_text(_toy_pdb(), encoding="utf-8")
    ligand.write_text(_toy_sdf(), encoding="utf-8")

    water_map = build_crystal_water_site_map(
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
                source="toy_fpocket",
                features={},
            ),
        ),
    )
    merged = merge_site_maps([water_map, hydrophobic_map], merge_distance=0.4, max_sites=10)
    site_map_path = output_dir / "toy_site_map.json"
    merged.write_json(site_map_path)

    ligand_atoms = read_ligand_atoms(ligand)
    pairs = build_atom_site_pairs(ligand_atoms, merged, config=PairBuildConfig(negative_ratio=2, split="train"))
    pair_path = output_dir / "toy_atom_site_pairs.jsonl"
    write_pairs_jsonl(pair_path, pairs)

    metrics = evaluate_site_matching(ligand_atoms, merged)
    metric_json = output_dir / "toy_site_metrics.json"
    metric_csv = output_dir / "toy_site_metrics.csv"
    write_metrics_outputs(metrics, output_json=metric_json, output_csv=metric_csv)

    quality_csv = output_dir / "toy_quality.csv"
    with quality_csv.open("wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "heavy_atom_count", "note"])
        writer.writeheader()
        writer.writerow({"file": str(ligand), "heavy_atom_count": len(ligand_atoms), "note": "toy ligand"})

    failure = analyze_site_metric_rows([asdict(metrics)])
    failure_json = output_dir / "toy_failure_analysis.json"
    failure_json.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")

    write_pymol_script(merged, output=output_dir / "toy_sites.pml", receptor=protein, ligand=ligand)
    plot_site_map(merged, output_dir / "toy_site_map.png")
    generate_html_report(
        output=output_dir / "toy_report.html",
        title="ESField Toy Pipeline Report",
        metric_files=[metric_json, failure_json],
        notes="这是无卡 toy pipeline，用于验证代码接口，不代表真实实验结果。",
    )
    print(f"toy pipeline completed: {output_dir}")


def _pdb_atom(record: str, serial: int, name: str, res: str, chain: str, resid: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"{record:<6}{serial:5d} {name:^4} {res:>3} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2}\n"
    )


def _toy_pdb() -> str:
    return "".join(
        [
            _pdb_atom("ATOM", 1, "N", "ALA", "A", 1, 0.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 2, "O", "ALA", "A", 1, 0.0, -2.8, 0.0, "O"),
            _pdb_atom("ATOM", 3, "C", "VAL", "A", 2, -2.0, 2.9, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "VAL", "A", 2, -2.7, 0.0, 0.0, "C"),
            _pdb_atom("HETATM", 5, "O", "HOH", "A", 10, 1.0, 0.0, 0.0, "O"),
            _pdb_atom("HETATM", 6, "O", "HOH", "A", 11, -2.0, 0.0, 0.0, "O"),
        ]
    )


def _toy_sdf() -> str:
    return """toy
  ESField

  3  0  0  0  0  0            999 V2000
    1.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""


if __name__ == "__main__":
    main()
