#!/usr/bin/env python3
"""Build training pairs from PDBbind water-included site maps.

Reads all built site maps (with water site types), extracts ligands from PDBbind,
builds atom-site pairs with train/valid split.

Usage:
  python scripts/build_pdbbind_pairs.py \
    --site-maps-dir experiments/pdbbind_water_sites/site_maps \
    --pdbbind-root /root/autodl-tmp/data/PDB/P-L \
    --output-dir experiments/pdbbind_water_sites
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from site_detection.site_schema import SiteMap
from data.build_atom_site_pairs import build_atom_site_pairs, PairBuildConfig
from data.atom_site_schema import write_pairs_jsonl
from utils.structure_io import read_ligand_atoms


def find_pdbbind_pocket_dir(pdb_id: str, pdbbind_root: Path) -> Path | None:
    """Find a PDBbind pocket directory by PDB ID."""
    for year_dir in sorted(pdbbind_root.iterdir()):
        if not year_dir.is_dir():
            continue
        candidate = year_dir / pdb_id
        if candidate.is_dir():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-maps-dir", default="experiments/pdbbind_water_sites/site_maps")
    parser.add_argument("--pdbbind-root", default="/root/autodl-tmp/data/PDB/P-L")
    parser.add_argument("--output-dir", default="experiments/pdbbind_water_sites")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    site_maps_dir = Path(args.site_maps_dir)
    map_files = sorted(site_maps_dir.glob("*_site_map.json"))
    print(f"Found {len(map_files)} site maps")

    all_train_pairs = []
    all_valid_pairs = []
    n_processed = 0
    n_skipped = 0
    split_idx = int(len(map_files) * args.train_ratio)

    site_type_counts = {"high_energy_water": 0, "stable_water": 0, "hydrophobic_cavity": 0}

    for i, map_file in enumerate(map_files):
        pdb_id = map_file.stem.replace("_site_map", "")

        try:
            site_map = SiteMap.read_json(str(map_file))
        except Exception as e:
            print(f"[{i+1}/{len(map_files)}] {pdb_id}: SKIP (bad site map: {e})")
            n_skipped += 1
            continue

        # Count site types
        for site in site_map.sites:
            if site.site_type in site_type_counts:
                site_type_counts[site.site_type] += 1

        # Find ligand
        pocket_dir = find_pdbbind_pocket_dir(pdb_id, Path(args.pdbbind_root))
        if pocket_dir is None:
            print(f"[{i+1}/{len(map_files)}] {pdb_id}: SKIP (PBDbind dir not found)")
            n_skipped += 1
            continue

        lig_sdf = pocket_dir / f"{pdb_id}_ligand.sdf"
        if not lig_sdf.exists():
            print(f"[{i+1}/{len(map_files)}] {pdb_id}: SKIP (no ligand SDF)")
            n_skipped += 1
            continue

        try:
            ligand_atoms = read_ligand_atoms(str(lig_sdf))
        except Exception as e:
            print(f"[{i+1}/{len(map_files)}] {pdb_id}: SKIP (ligand read error: {e})")
            n_skipped += 1
            continue

        if len(ligand_atoms) < 2:
            print(f"[{i+1}/{len(map_files)}] {pdb_id}: SKIP (< 2 heavy atoms)")
            n_skipped += 1
            continue

        # Build pairs
        split = "train" if i < split_idx else "valid"
        try:
            pairs = build_atom_site_pairs(
                ligand_atoms, site_map,
                config=PairBuildConfig(split=split, negative_ratio=args.negative_ratio, random_seed=args.seed),
            )
        except Exception as e:
            print(f"[{i+1}/{len(map_files)}] {pdb_id}: SKIP (pair error: {e})")
            n_skipped += 1
            continue

        if split == "train":
            all_train_pairs.extend(pairs)
        else:
            all_valid_pairs.extend(pairs)

        n_pos = sum(1 for p in pairs if p.label == 1)
        n_neg = sum(1 for p in pairs if p.label == 0)
        n_processed += 1
        print(f"[{i+1}/{len(map_files)}] {pdb_id}: {len(pairs)} pairs (+{n_pos}/-{n_neg}) [{split}]")

    # Write
    train_path = output_dir / "train_pairs.jsonl"
    valid_path = output_dir / "valid_pairs.jsonl"
    write_pairs_jsonl(train_path, all_train_pairs)
    write_pairs_jsonl(valid_path, all_valid_pairs)

    total = len(all_train_pairs) + len(all_valid_pairs)
    n_pos_total = sum(1 for p in all_train_pairs + all_valid_pairs if p.label == 1)

    print(f"\nDone. {n_processed} processed, {n_skipped} skipped.")
    print(f"Train: {len(all_train_pairs)} pairs ({train_path})")
    print(f"Valid: {len(all_valid_pairs)} pairs ({valid_path})")
    print(f"Total: {total} pairs (pos={n_pos_total}, neg={total - n_pos_total})")
    print(f"Site type distribution: {json.dumps(site_type_counts)}")


if __name__ == "__main__":
    main()
