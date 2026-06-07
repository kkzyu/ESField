"""Parse fpocket output into hydrophobic cavity sites."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from site_detection.site_schema import Site, SiteMap
from utils.geometry import centroid, distance
from utils.structure_io import read_pdb_atoms


@dataclass(frozen=True)
class FpocketParseConfig:
    radius_min: float = 0.8
    radius_max: float = 3.5
    max_sites: int = 20
    min_alpha_spheres: int = 3


def parse_fpocket_site_map(
    fpocket_dir: str | Path,
    *,
    protein_id: str,
    ligand_id: str = "unknown_ligand",
    pocket_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    config: FpocketParseConfig | None = None,
) -> SiteMap:
    cfg = config or FpocketParseConfig()
    root = Path(fpocket_dir)
    pocket_files = sorted(root.glob("**/pocket*_atm.pdb"))
    if not pocket_files:
        pocket_files = sorted(root.glob("**/pocket*.pdb"))
    info_by_index = _read_info_files(root)

    sites: list[Site] = []
    for pocket_file in pocket_files:
        atoms = read_pdb_atoms(pocket_file)
        if len(atoms) < cfg.min_alpha_spheres:
            continue
        center = centroid(atom.coord for atom in atoms)
        max_radius = max(distance(center, atom.coord) for atom in atoms)
        radius = min(cfg.radius_max, max(cfg.radius_min, max_radius))
        pocket_index = _extract_pocket_index(pocket_file.name)
        info = info_by_index.get(pocket_index, {})
        score = float(info.get("score", len(atoms)))
        hydrophobicity = float(info.get("hydrophobicity_score", info.get("hydrophobicity", 0.0)))
        confidence = min(1.0, max(0.2, 0.35 + 0.03 * len(atoms) + 0.05 * max(0.0, hydrophobicity)))
        sites.append(
            Site(
                site_id=len(sites),
                site_type="hydrophobic_cavity",
                center=center,
                radius=radius,
                score=score,
                confidence=confidence,
                source="fpocket",
                features={
                    "pocket_file": str(pocket_file),
                    "pocket_index": pocket_index,
                    "alpha_sphere_count": len(atoms),
                    "hydrophobicity_score": hydrophobicity,
                },
            )
        )

    sites.sort(key=lambda site: (site.confidence, site.score), reverse=True)
    selected = tuple(_renumber_sites(sites[: cfg.max_sites]))
    return SiteMap(
        protein_id=protein_id,
        ligand_id=ligand_id,
        pocket_center=pocket_center,
        coordinate_frame="original_pdb_coordinates",
        sites=selected,
    )


def _read_info_files(root: Path) -> dict[int, dict[str, float]]:
    info: dict[int, dict[str, float]] = {}
    for info_file in root.glob("**/*info*.txt"):
        current_index: int | None = None
        for line in info_file.read_text(encoding="utf-8", errors="replace").splitlines():
            pocket_match = re.search(r"Pocket\s+(\d+)", line, flags=re.IGNORECASE)
            if pocket_match:
                current_index = int(pocket_match.group(1))
                info.setdefault(current_index, {})
                continue
            if current_index is None or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_")
            value_match = re.search(r"-?\d+(?:\.\d+)?", raw_value)
            if value_match:
                info[current_index][normalized_key] = float(value_match.group(0))
    return info


def _extract_pocket_index(name: str) -> int:
    match = re.search(r"pocket(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _renumber_sites(sites: list[Site]) -> list[Site]:
    return [
        Site(
            site_id=index,
            site_type=site.site_type,
            center=site.center,
            radius=site.radius,
            score=site.score,
            confidence=site.confidence,
            source=site.source,
            features=site.features,
        )
        for index, site in enumerate(sites)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse fpocket output into ESField hydrophobic sites.")
    parser.add_argument("--fpocket-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--ligand-id", default="unknown_ligand")
    parser.add_argument("--pocket-center", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output}")

    site_map = parse_fpocket_site_map(
        args.fpocket_dir,
        protein_id=args.protein_id,
        ligand_id=args.ligand_id,
        pocket_center=tuple(args.pocket_center),
        config=FpocketParseConfig(max_sites=args.max_sites),
    )
    print(f"parsed {len(site_map.sites)} fpocket hydrophobic sites for {site_map.protein_id}")
    if not args.dry_run:
        site_map.write_json(output)


if __name__ == "__main__":
    main()

