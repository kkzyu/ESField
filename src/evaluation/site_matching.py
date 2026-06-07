"""Site-specific ESField metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from site_detection.site_schema import SiteMap
from utils.chemistry import is_compatible_atom_site
from utils.geometry import distance, gaussian_weight
from utils.structure_io import StructureAtom, read_ligand_atoms


@dataclass(frozen=True)
class SiteMatchingMetrics:
    site_occupancy_rate: float
    correct_atom_site_matching_rate: float
    high_energy_water_replacement_rate: float
    stable_water_preservation_penalty: float
    hydrophobic_cavity_filling_score: float
    occupied_sites: int
    compatible_occupied_sites: int
    total_sites: int


def evaluate_site_matching(ligand_atoms: list[StructureAtom], site_map: SiteMap) -> SiteMatchingMetrics:
    total_sites = len(site_map.sites)
    if total_sites == 0:
        return SiteMatchingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    occupied = 0
    compatible_occupied = 0
    high_energy_total = 0
    high_energy_occupied = 0
    stable_total = 0
    stable_incompatible = 0
    hydrophobic_scores: list[float] = []

    for site in site_map.sites:
        atom_distances = [(atom, distance(atom.coord, site.center)) for atom in ligand_atoms]
        within = [(atom, d) for atom, d in atom_distances if d <= site.radius]
        compatible_within = [
            (atom, d)
            for atom, d in within
            if is_compatible_atom_site(atom.atom_type, atom.atomic_number, site.site_type)
        ]
        if within:
            occupied += 1
        if compatible_within:
            compatible_occupied += 1

        if site.site_type == "high_energy_water":
            high_energy_total += 1
            if compatible_within:
                high_energy_occupied += 1
        elif site.site_type == "stable_water":
            stable_total += 1
            if within and not compatible_within:
                stable_incompatible += 1
        elif site.site_type == "hydrophobic_cavity":
            best = 0.0
            for atom, d in atom_distances:
                if is_compatible_atom_site(atom.atom_type, atom.atomic_number, site.site_type):
                    best = max(best, gaussian_weight(d, max(site.radius, 0.1)))
            hydrophobic_scores.append(best)

    return SiteMatchingMetrics(
        site_occupancy_rate=occupied / total_sites,
        correct_atom_site_matching_rate=compatible_occupied / occupied if occupied else 0.0,
        high_energy_water_replacement_rate=high_energy_occupied / high_energy_total if high_energy_total else 0.0,
        stable_water_preservation_penalty=stable_incompatible / stable_total if stable_total else 0.0,
        hydrophobic_cavity_filling_score=sum(hydrophobic_scores) / len(hydrophobic_scores) if hydrophobic_scores else 0.0,
        occupied_sites=occupied,
        compatible_occupied_sites=compatible_occupied,
        total_sites=total_sites,
    )


def write_metrics_outputs(metrics: SiteMatchingMetrics, *, output_json: str | Path | None, output_csv: str | Path | None) -> None:
    row = asdict(metrics)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_csv:
        target = Path(output_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ESField site-specific ligand metrics.")
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    metrics = evaluate_site_matching(read_ligand_atoms(args.ligand), SiteMap.read_json(args.site_map))
    print(json.dumps(asdict(metrics), indent=2, ensure_ascii=False))
    write_metrics_outputs(metrics, output_json=args.output_json, output_csv=args.output_csv)


if __name__ == "__main__":
    main()

