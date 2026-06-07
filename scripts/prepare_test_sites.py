#!/usr/bin/env python3
"""Build site maps for test pockets + generate random/shuffled controls.

Usage:
  python scripts/prepare_test_sites.py
"""

from __future__ import annotations

import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from site_detection.site_schema import Site, SiteMap
from site_detection.parse_fpocket import parse_fpocket_site_map, FpocketParseConfig
from site_detection.build_crystal_water_sites import build_crystal_water_site_map, CrystalWaterConfig
from site_detection.merge_sites import merge_site_maps
from utils.structure_io import read_pdb_atoms
from utils.geometry import centroid, distance
import subprocess


def run_fpocket(pdb_path: Path) -> Path | None:
    pdb_dir = pdb_path.resolve().parent
    expected_out = pdb_dir / f"{pdb_path.stem}_out"
    if expected_out.exists() and (expected_out / "pockets").exists():
        return expected_out
    try:
        subprocess.run(["fpocket", "-f", str(pdb_path.resolve())],
                       capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return expected_out if expected_out.exists() else None


def build_site_map(protein_pdb: Path, pocket_pdb: Path, lig_sdf: Path | None,
                   pdb_id: str, max_sites: int = 20) -> SiteMap:
    """Build combined water+hydrophobic site map for one pocket."""
    atoms = read_pdb_atoms(str(pocket_pdb))
    if atoms:
        pc = centroid(a.coord for a in atoms)
    else:
        pc = (0.0, 0.0, 0.0)

    # fpocket hydrophobic sites (use pocket PDB for speed)
    fpocket_out = run_fpocket(pocket_pdb)
    if fpocket_out is None:
        fpocket_map = SiteMap(
            protein_id=pdb_id, ligand_id=pdb_id,
            pocket_center=pc, coordinate_frame="original_pdb_coordinates", sites=())
    else:
        fpocket_map = parse_fpocket_site_map(
            fpocket_out, protein_id=pdb_id, ligand_id=pdb_id,
            pocket_center=pc, config=FpocketParseConfig(max_sites=10))

    # Crystal water sites
    water_map = build_crystal_water_site_map(
        str(protein_pdb),
        ligand_path=str(lig_sdf) if lig_sdf else None,
        protein_id=pdb_id, ligand_id=pdb_id,
        pocket_center=pc,
        config=CrystalWaterConfig(pocket_radius=10.0, max_sites=15))

    return merge_site_maps([fpocket_map, water_map], merge_distance=1.0, max_sites=max_sites)


def generate_random_site_map(site_map: SiteMap, protein_pdb: Path, pocket_pdb: Path,
                              seed: int = 42) -> SiteMap:
    """Generate a site map with random coordinates (same type distribution, random positions)."""
    rng = random.Random(seed)
    atoms = read_pdb_atoms(str(pocket_pdb))
    if not atoms:
        return site_map

    # Compute pocket bounding box
    coords = [a.coord for a in atoms]
    mins = [min(c[i] for c in coords) for i in range(3)]
    maxs = [max(c[i] for c in coords) for i in range(3)]
    pad = 2.0

    random_sites = []
    for i, site in enumerate(site_map.sites):
        center = tuple(rng.uniform(mins[j] - pad, maxs[j] + pad) for j in range(3))
        random_sites.append(Site(
            site_id=i, site_type=site.site_type, center=center,
            radius=site.radius, score=site.score, confidence=site.confidence,
            source="random_coordinates",
            features={"original_site_id": site.site_id}))

    return SiteMap(
        protein_id=site_map.protein_id, ligand_id=site_map.ligand_id,
        pocket_center=site_map.pocket_center,
        coordinate_frame="random_coordinates",
        sites=tuple(random_sites))


def generate_shuffled_site_map(site_map: SiteMap, seed: int = 42) -> SiteMap:
    """Generate a site map with shuffled site types (same coordinates, random types)."""
    rng = random.Random(seed)
    types = [s.site_type for s in site_map.sites]
    rng.shuffle(types)

    shuffled_sites = []
    for i, site in enumerate(site_map.sites):
        shuffled_sites.append(Site(
            site_id=i, site_type=types[i], center=site.center,
            radius=site.radius, score=site.score, confidence=site.confidence,
            source="shuffled_types",
            features={"original_type": site.site_type}))

    return SiteMap(
        protein_id=site_map.protein_id, ligand_id=site_map.ligand_id,
        pocket_center=site_map.pocket_center,
        coordinate_frame=site_map.coordinate_frame,
        sites=tuple(shuffled_sites))


def main():
    test_pockets = json.load(open(ROOT / "experiments/pdbbind_water_sites/test_pockets.json"))

    base_dir = ROOT / "experiments/pdbbind_water_sites/test_sites"
    for subdir in ["correct", "random", "shuffled"]:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    n_built = 0
    for i, pocket in enumerate(test_pockets):
        pdb_id = pocket["pdb_id"]
        pocket_dir = Path(pocket["dir"])
        protein_pdb = pocket_dir / f"{pdb_id}_protein.pdb"
        pocket_pdb = pocket_dir / f"{pdb_id}_pocket.pdb"
        lig_sdf = pocket_dir / f"{pdb_id}_ligand.sdf"

        correct_path = base_dir / "correct" / f"{pdb_id}_site_map.json"
        random_path = base_dir / "random" / f"{pdb_id}_site_map.json"
        shuffled_path = base_dir / "shuffled" / f"{pdb_id}_site_map.json"

        if correct_path.exists() and random_path.exists() and shuffled_path.exists():
            sm = SiteMap.read_json(str(correct_path))
            n_types = len(set(s.site_type for s in sm.sites))
            print(f"[{i+1}/{len(test_pockets)}] {pdb_id}: cached ({len(sm.sites)} sites, {n_types} types)")
            n_built += 1
            continue

        try:
            site_map = build_site_map(protein_pdb, pocket_pdb, lig_sdf, pdb_id)
            site_map.write_json(str(correct_path))

            rand_map = generate_random_site_map(site_map, protein_pdb, pocket_pdb)
            rand_map.write_json(str(random_path))

            shuf_map = generate_shuffled_site_map(site_map)
            shuf_map.write_json(str(shuffled_path))

            n_types = len(set(s.site_type for s in site_map.sites))
            print(f"[{i+1}/{len(test_pockets)}] {pdb_id}: {len(site_map.sites)} sites ({n_types} types)")
            n_built += 1
        except Exception as e:
            print(f"[{i+1}/{len(test_pockets)}] {pdb_id}: ERROR {e}")

    print(f"\nBuilt {n_built}/{len(test_pockets)} test site maps")


if __name__ == "__main__":
    main()
