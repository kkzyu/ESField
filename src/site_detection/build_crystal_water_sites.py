"""Build MVP water sites from crystal waters in a receptor PDB."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from site_detection.site_schema import Site, SiteMap
from utils.chemistry import is_hydrophobic_element, is_polar_element, is_water_residue
from utils.geometry import centroid, distance
from utils.structure_io import StructureAtom, read_ligand_atoms, read_pdb_atoms


@dataclass(frozen=True)
class CrystalWaterConfig:
    pocket_radius: float = 10.0
    water_radius: float = 1.4
    protein_clash_distance: float = 1.6
    hbond_distance: float = 3.5
    hydrophobic_distance: float = 4.0
    stable_hbond_min: int = 2
    high_energy_hbond_max: int = 1
    high_energy_hydrophobic_min: int = 3
    max_sites: int = 20


def build_crystal_water_site_map(
    protein_pdb: str | Path,
    *,
    ligand_path: str | Path | None = None,
    protein_id: str | None = None,
    ligand_id: str | None = None,
    pocket_center: tuple[float, float, float] | None = None,
    config: CrystalWaterConfig | None = None,
) -> SiteMap:
    cfg = config or CrystalWaterConfig()
    atoms = read_pdb_atoms(protein_pdb)
    waters = _group_water_oxygens(atoms)
    protein_atoms = [atom for atom in atoms if not is_water_residue(atom.residue_name)]
    center = _resolve_pocket_center(ligand_path, pocket_center, protein_atoms)

    if not protein_atoms:
        raise ValueError("no protein atoms found in PDB; cannot build water sites")
    candidates: list[Site] = []
    for water_index, water in enumerate(waters):
        if distance(water.coord, center) > cfg.pocket_radius:
            continue
        nearest_protein = min((distance(water.coord, atom.coord) for atom in protein_atoms))
        if nearest_protein <= cfg.protein_clash_distance:
            continue
        hbond_count = sum(
            1
            for atom in protein_atoms
            if is_polar_element(atom.element) and distance(water.coord, atom.coord) <= cfg.hbond_distance
        )
        hydrophobic_count = sum(
            1
            for atom in protein_atoms
            if is_hydrophobic_element(atom.element)
            and distance(water.coord, atom.coord) <= cfg.hydrophobic_distance
        )
        site_type, score, confidence = _classify_water_site(
            hbond_count=hbond_count,
            hydrophobic_count=hydrophobic_count,
            cfg=cfg,
        )
        candidates.append(
            Site(
                site_id=water_index,
                site_type=site_type,
                center=water.coord,
                radius=cfg.water_radius,
                score=score,
                confidence=confidence,
                source="crystal_water_rule",
                features={
                    "residue_name": water.residue_name,
                    "residue_id": water.residue_id,
                    "chain_id": water.chain_id,
                    "hbond_count": hbond_count,
                    "hydrophobic_contact_count": hydrophobic_count,
                    "nearest_protein_distance": round(nearest_protein, 4),
                },
            )
        )

    candidates.sort(key=lambda site: (site.confidence, abs(site.score)), reverse=True)
    selected = tuple(_renumber_sites(candidates[: cfg.max_sites]))
    return SiteMap(
        protein_id=protein_id or Path(protein_pdb).stem,
        ligand_id=ligand_id or (Path(ligand_path).stem if ligand_path else "unknown_ligand"),
        pocket_center=center,
        coordinate_frame="original_pdb_coordinates",
        sites=selected,
    )


def _group_water_oxygens(atoms: list[StructureAtom]) -> list[StructureAtom]:
    grouped: dict[tuple[str, str, str], list[StructureAtom]] = defaultdict(list)
    for atom in atoms:
        if is_water_residue(atom.residue_name) and atom.element == "O":
            grouped[(atom.chain_id, atom.residue_id, atom.residue_name)].append(atom)
    waters: list[StructureAtom] = []
    for group in grouped.values():
        waters.append(group[0])
    return waters


def _resolve_pocket_center(
    ligand_path: str | Path | None,
    pocket_center: tuple[float, float, float] | None,
    protein_atoms: list[StructureAtom],
) -> tuple[float, float, float]:
    if pocket_center is not None:
        return pocket_center
    if ligand_path is not None:
        ligand_atoms = read_ligand_atoms(ligand_path)
        if not ligand_atoms:
            raise ValueError(f"ligand file has no heavy atoms: {ligand_path}")
        return centroid(atom.coord for atom in ligand_atoms)
    if protein_atoms:
        return centroid(atom.coord for atom in protein_atoms)
    raise ValueError("cannot resolve pocket center without ligand or protein atoms")


def _classify_water_site(
    *,
    hbond_count: int,
    hydrophobic_count: int,
    cfg: CrystalWaterConfig,
) -> tuple[str, float, float]:
    if hbond_count >= cfg.stable_hbond_min:
        score = -float(hbond_count)
        confidence = min(1.0, 0.55 + 0.15 * hbond_count)
        return "stable_water", score, confidence
    if hbond_count <= cfg.high_energy_hbond_max or hydrophobic_count >= cfg.high_energy_hydrophobic_min:
        score = 0.8 + 0.4 * hydrophobic_count - 0.2 * hbond_count
        confidence = min(1.0, 0.45 + 0.1 * hydrophobic_count + 0.1 * (cfg.high_energy_hbond_max - hbond_count + 1))
        return "high_energy_water", score, confidence
    return "stable_water", -0.5, 0.5


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
    parser = argparse.ArgumentParser(description="Build ESField crystal-water site map from a receptor PDB.")
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--ligand", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protein-id", default=None)
    parser.add_argument("--ligand-id", default=None)
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output}")

    config = CrystalWaterConfig(pocket_radius=args.pocket_radius, max_sites=args.max_sites)
    site_map = build_crystal_water_site_map(
        args.protein_pdb,
        ligand_path=args.ligand,
        protein_id=args.protein_id,
        ligand_id=args.ligand_id,
        config=config,
    )
    print(f"built {len(site_map.sites)} crystal-water sites for {site_map.protein_id}")
    if not args.dry_run:
        site_map.write_json(output)


if __name__ == "__main__":
    main()

