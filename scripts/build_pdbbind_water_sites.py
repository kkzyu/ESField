#!/usr/bin/env python3
"""Build water-included site maps from PDBbind pockets.

For each selected pocket:
1. Run fpocket to get hydrophobic cavity sites
2. Detect crystal water sites (high_energy_water / stable_water)
3. Merge into combined site map with all three site types

Usage:
  python scripts/build_pdbbind_water_sites.py \
    --pdbbind-root /root/autodl-tmp/data/PDB/P-L \
    --output-dir experiments/pdbbind_water_sites \
    --max-atoms 2000 --n-pockets 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from site_detection.site_schema import Site, SiteMap
from site_detection.parse_fpocket import parse_fpocket_site_map, FpocketParseConfig
from site_detection.build_crystal_water_sites import (
    build_crystal_water_site_map,
    CrystalWaterConfig,
)
from site_detection.merge_sites import merge_site_maps
from utils.structure_io import read_ligand_atoms, read_pdb_atoms
from utils.geometry import centroid


def count_atoms(pdb_path: Path) -> int:
    n = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                n += 1
    return n


def run_fpocket(receptor_pdb: Path) -> Path | None:
    pdb_dir = receptor_pdb.resolve().parent
    expected_out = pdb_dir / f"{receptor_pdb.stem}_out"
    if expected_out.exists() and (expected_out / "pockets").exists():
        return expected_out
    try:
        subprocess.run(
            ["fpocket", "-f", str(receptor_pdb.resolve())],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    fpocket failed: {e}")
        return None
    if expected_out.exists():
        return expected_out
    return None


def select_pockets(
    pdbbind_root: Path, max_atoms: int, n_pockets: int, seed: int = 42
) -> list[dict]:
    """Scan and select pockets with < max_atoms atoms, preferring ones with crystal water."""
    import random
    random.seed(seed)

    candidates = []
    for year_dir in sorted(pdbbind_root.iterdir()):
        if not year_dir.is_dir():
            continue
        for pdb_dir in sorted(year_dir.iterdir()):
            if not pdb_dir.is_dir():
                continue
            pocket_pdb = pdb_dir / f"{pdb_dir.name}_pocket.pdb"
            protein_pdb = pdb_dir / f"{pdb_dir.name}_protein.pdb"
            lig_sdf = pdb_dir / f"{pdb_dir.name}_ligand.sdf"
            if not pocket_pdb.exists() or not protein_pdb.exists():
                continue

            n_atoms = count_atoms(pocket_pdb)
            if n_atoms >= max_atoms:
                continue

            n_waters = 0
            with open(protein_pdb) as f:
                for line in f:
                    if line.startswith("HETATM") and "HOH" in line[17:20]:
                        n_waters += 1

            has_ligand = lig_sdf.exists()
            candidates.append({
                "pdb_id": pdb_dir.name,
                "dir": str(pdb_dir),
                "protein_pdb": str(protein_pdb),
                "pocket_pdb": str(pocket_pdb),
                "ligand_sdf": str(lig_sdf) if has_ligand else None,
                "has_ligand": has_ligand,
                "n_atoms": n_atoms,
                "n_waters": n_waters,
            })

    # Sort: prefer pockets with water and ligand
    candidates.sort(key=lambda c: (c["n_waters"] > 0, c["has_ligand"], c["n_waters"]), reverse=True)
    selected = candidates[:n_pockets]
    print(f"Selected {len(selected)} from {len(candidates)} eligible pockets")
    with_water = sum(1 for c in selected if c["n_waters"] > 0)
    with_ligand = sum(1 for c in selected if c["ligand_sdf"])
    print(f"  with crystal water: {with_water}")
    print(f"  with ligand: {with_ligand}")
    return selected


def get_pocket_center_from_pocket_pdb(pocket_pdb: Path) -> tuple[float, float, float]:
    atoms = read_pdb_atoms(str(pocket_pdb))
    if not atoms:
        return (0.0, 0.0, 0.0)
    return centroid(atom.coord for atom in atoms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbbind-root", default="/root/autodl-tmp/data/PDB/P-L")
    parser.add_argument("--output-dir", default="experiments/pdbbind_water_sites")
    parser.add_argument("--max-atoms", type=int, default=2000)
    parser.add_argument("--n-pockets", type=int, default=300)
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--fpocket-max-sites", type=int, default=10)
    parser.add_argument("--water-max-sites", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    site_maps_dir = output_dir / "site_maps"
    site_maps_dir.mkdir(exist_ok=True)
    summary_path = output_dir / "build_summary.json"

    # Select pockets
    pockets = select_pockets(
        Path(args.pdbbind_root), args.max_atoms, args.n_pockets, args.seed
    )
    with open(output_dir / "selected_pockets.json", "w") as f:
        json.dump(pockets, f, indent=2)

    results = []
    start_time = time.time()

    for i, pocket in enumerate(pockets):
        pdb_id = pocket["pdb_id"]
        protein_pdb = Path(pocket["protein_pdb"])
        pocket_pdb = Path(pocket["pocket_pdb"])
        lig_sdf = Path(pocket["ligand_sdf"]) if pocket["ligand_sdf"] else None
        site_map_path = site_maps_dir / f"{pdb_id}_site_map.json"

        elapsed = time.time() - start_time
        eta = (elapsed / max(i, 1)) * (len(pockets) - i) if i > 0 else 0
        status = f"[{i+1}/{len(pockets)}] {pdb_id} ({pocket['n_atoms']} atoms, {pocket['n_waters']} waters)"

        if args.skip_existing and site_map_path.exists():
            try:
                sm = SiteMap.read_json(str(site_map_path))
                n_hydro = sum(1 for s in sm.sites if s.site_type == "hydrophobic_cavity")
                n_w = sum(1 for s in sm.sites if s.site_type in ("high_energy_water", "stable_water"))
                results.append({"pdb_id": pdb_id, "n_sites": len(sm.sites),
                                "n_hydrophobic": n_hydro, "n_water": n_w,
                                "status": "cached"})
                print(f"{status}: cached ({len(sm.sites)} sites, {n_w} water)")
                continue
            except Exception:
                pass

        try:
            pc = get_pocket_center_from_pocket_pdb(pocket_pdb)

            # 1. fpocket hydrophobic sites (use pocket PDB for speed, not full protein)
            fpocket_out = run_fpocket(pocket_pdb)
            if fpocket_out is None:
                fpocket_map = SiteMap(
                    protein_id=pdb_id, ligand_id=pdb_id,
                    pocket_center=pc, coordinate_frame="original_pdb_coordinates",
                    sites=(),
                )
            else:
                fpocket_map = parse_fpocket_site_map(
                    fpocket_out, protein_id=pdb_id, ligand_id=pdb_id,
                    pocket_center=pc,
                    config=FpocketParseConfig(max_sites=args.fpocket_max_sites),
                )

            # 2. Crystal water sites
            water_map = build_crystal_water_site_map(
                str(protein_pdb),
                ligand_path=str(lig_sdf) if lig_sdf else None,
                protein_id=pdb_id, ligand_id=pdb_id,
                pocket_center=pc,
                config=CrystalWaterConfig(
                    pocket_radius=10.0,
                    max_sites=args.water_max_sites,
                ),
            )

            # 3. Merge
            all_maps = [fpocket_map, water_map]
            merged = merge_site_maps(
                all_maps, merge_distance=1.0, max_sites=args.max_sites
            )
            merged.write_json(str(site_map_path))

            n_hydro = sum(1 for s in merged.sites if s.site_type == "hydrophobic_cavity")
            n_w = sum(1 for s in merged.sites if s.site_type in ("high_energy_water", "stable_water"))
            results.append({"pdb_id": pdb_id, "n_sites": len(merged.sites),
                            "n_hydrophobic": n_hydro, "n_water": n_w,
                            "status": "ok"})
            print(f"{status}: {len(merged.sites)} sites (hydro={n_hydro}, water={n_w}) ETA={eta:.0f}s")

        except Exception as e:
            results.append({"pdb_id": pdb_id, "n_sites": 0, "status": f"error: {e}"})
            print(f"{status}: ERROR {e}")

    # Summary
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    n_water = sum(1 for r in results if r.get("n_water", 0) > 0)
    print(f"\nDone. {n_ok}/{len(pockets)} successful, {n_water} with water sites")
    print(f"Site maps: {site_maps_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
