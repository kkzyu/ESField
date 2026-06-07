#!/usr/bin/env python3
"""Batch process PAFlow test set pockets: build site maps + atom-site pairs.

Usage:
  python scripts/batch_build_pairs.py --n-pockets 20 --output-dir experiments/potential_training
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from site_detection.site_schema import SiteMap
from data.build_atom_site_pairs import build_atom_site_pairs, PairBuildConfig
from data.atom_site_schema import write_pairs_jsonl
from utils.structure_io import read_ligand_atoms
from utils.geometry import centroid


def fpocket_run(receptor_pdb: Path) -> Path:
    """Run fpocket on a receptor PDB, return the fpocket output dir."""
    pdb_dir = receptor_pdb.resolve().parent
    expected_out = pdb_dir / f"{receptor_pdb.stem}_out"
    if expected_out.exists() and (expected_out / "pockets").exists():
        return expected_out
    cmd = ["fpocket", "-f", str(receptor_pdb.resolve())]
    subprocess.run(cmd, capture_output=True, text=True)
    if not expected_out.exists():
        raise FileNotFoundError(f"fpocket output not found: {expected_out}")
    return expected_out


def build_site_map_fpocket_only(protein_pdb: Path, protein_id: str) -> SiteMap:
    """Build site map from fpocket only (no crystal water)."""
    from site_detection.parse_fpocket import parse_fpocket_site_map, FpocketParseConfig
    from site_detection.merge_sites import merge_site_maps

    fpocket_out = fpocket_run(protein_pdb)
    atoms = []
    with open(protein_pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    atoms.append((x, y, z))
                except ValueError:
                    continue
    if atoms:
        pc = tuple(sum(c) / len(atoms) for c in zip(*atoms))
    else:
        pc = (0.0, 0.0, 0.0)

    fpocket_map = parse_fpocket_site_map(
        fpocket_out,
        protein_id=protein_id,
        ligand_id="docked",
        pocket_center=pc,
        config=FpocketParseConfig(max_sites=10),
    )
    merged = merge_site_maps([fpocket_map], merge_distance=1.0, max_sites=15)
    return merged




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pockets", type=int, default=20)
    parser.add_argument("--output-dir", default="experiments/potential_training")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--max-sites", type=int, default=15)
    parser.add_argument("--test-set-dir", default="/root/PAFlow-main/data/test_set")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    site_maps_dir = output_dir / "site_maps"
    site_maps_dir.mkdir(exist_ok=True)

    test_set = sorted(Path(args.test_set_dir).iterdir())
    pockets = [d for d in test_set if d.is_dir()][:args.n_pockets]

    all_train_pairs = []
    all_valid_pairs = []
    n_processed = 0
    n_skipped = 0
    split_idx = int(len(pockets) * args.train_ratio)

    for i, pocket_dir in enumerate(pockets):
        protein_id = pocket_dir.name
        # Find receptor PDB
        rec_files = sorted(pocket_dir.glob("*_rec.pdb"))
        if not rec_files:
            print(f"[{i+1}/{len(pockets)}] {protein_id}: SKIP (no receptor)")
            n_skipped += 1
            continue

        rec_pdb = rec_files[0]
        # Find ligand (prefer ligand.pdb, then SDF)
        lig_pdb = pocket_dir / "ligand.pdb"
        sdf_files = sorted(pocket_dir.glob("*.sdf"))
        lig_source = None
        ligand_atoms = None

        if lig_pdb.exists():
            try:
                ligand_atoms = read_ligand_atoms(str(lig_pdb))
                lig_source = "ligand.pdb"
            except Exception:
                pass

        if ligand_atoms is None and sdf_files:
            for sdf_f in sdf_files:
                try:
                    ligand_atoms = read_ligand_atoms(str(sdf_f))
                    if len(ligand_atoms) > 2:
                        lig_source = sdf_f.name
                        break
                except Exception:
                    continue

        if ligand_atoms is None or len(ligand_atoms) < 2:
            print(f"[{i+1}/{len(pockets)}] {protein_id}: SKIP (no valid ligand)")
            n_skipped += 1
            continue

        # Build site map
        site_map_path = site_maps_dir / f"{protein_id}_site_map.json"
        try:
            if site_map_path.exists():
                site_map = SiteMap.read_json(str(site_map_path))
            else:
                site_map = build_site_map_fpocket_only(rec_pdb, protein_id)
                site_map.write_json(str(site_map_path))
        except Exception as e:
            print(f"[{i+1}/{len(pockets)}] {protein_id}: SKIP (site map error: {e})")
            n_skipped += 1
            continue

        # Build pairs
        split = "train" if i < split_idx else "valid"
        try:
            pairs = build_atom_site_pairs(
                ligand_atoms, site_map,
                config=PairBuildConfig(split=split, negative_ratio=3),
            )
        except Exception as e:
            print(f"[{i+1}/{len(pockets)}] {protein_id}: SKIP (pair error: {e})")
            n_skipped += 1
            continue

        if split == "train":
            all_train_pairs.extend(pairs)
        else:
            all_valid_pairs.extend(pairs)

        n_pos = sum(1 for p in pairs if p.label == 1)
        n_neg = sum(1 for p in pairs if p.label == 0)
        print(f"[{i+1}/{len(pockets)}] {protein_id}: {len(pairs)} pairs (+{n_pos}/-{n_neg}) [{split}] from {lig_source}")

    # Write combined pair files
    train_path = output_dir / "train_pairs.jsonl"
    valid_path = output_dir / "valid_pairs.jsonl"
    write_pairs_jsonl(train_path, all_train_pairs)
    write_pairs_jsonl(valid_path, all_valid_pairs)

    print(f"\nDone. {n_processed} processed, {n_skipped} skipped.")
    print(f"Train: {len(all_train_pairs)} pairs ({train_path})")
    print(f"Valid: {len(all_valid_pairs)} pairs ({valid_path})")


if __name__ == "__main__":
    main()
