#!/usr/bin/env python3
"""ESField + PAFlow end-to-end pipeline script.

This script automates the complete pre-generation workflow:
  1. Build ESField site map from PDB protein (crystal water + fpocket)
  2. Prepare PAFlow pocket data for a given protein
  3. Generate molecules with PAFlow (baseline)
  4. Inject ESField site guidance into PAFlow sampling (guided)

Usage:
  # Step 1-2: Site map + pocket preparation (no GPU needed)
  python scripts/run_esfield_paflow_pipeline.py \
    --protein-pdb /path/to/protein.pdb \
    --ligand-sdf /path/to/ligand.sdf \
    --protein-id 1abc \
    --output-dir experiments/esfield_paflow/1abc \
    --prepare-only

  # Step 3-4: Generation (needs GPU + PAFlow pretrained models)
  python scripts/run_esfield_paflow_pipeline.py \
    --protein-pdb /path/to/protein.pdb \
    --volume 232.6 --area 229.1 \
    --protein-id 1abc \
    --output-dir experiments/esfield_paflow/1abc \
    --paflow-dir /root/PAFlow-main \
    --mode both
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from site_detection.build_crystal_water_sites import (
    CrystalWaterConfig,
    build_crystal_water_site_map,
)
from site_detection.merge_sites import merge_site_maps
from site_detection.parse_fpocket import FpocketParseConfig, parse_fpocket_site_map
from site_detection.site_schema import SiteMap
from utils.geometry import centroid
from utils.structure_io import read_pdb_atoms


def run_fpocket(pdb_path: Path, output_dir: Path) -> Path:
    """Run fpocket on a protein PDB and return the output directory path.

    fpocket always creates {pdb_stem}_out/ alongside the input PDB,
    regardless of the -o flag. We create a symlink in output_dir for
    reproducible access, and return the actual fpocket output path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = pdb_path.resolve().parent
    expected_out = pdb_dir / f"{pdb_path.stem}_out"

    if not expected_out.exists():
        cmd = ["fpocket", "-f", str(pdb_path.resolve())]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"fpocket failed: {result.stderr}")

    if not expected_out.exists():
        raise FileNotFoundError(f"fpocket output directory not found: {expected_out}")

    # Create symlink in output_dir for tracking
    link_path = output_dir / expected_out.name
    if not link_path.exists():
        link_path.symlink_to(expected_out)

    return expected_out


def build_full_site_map(
    protein_pdb: Path,
    ligand_sdf: Path | None = None,
    *,
    protein_id: str,
    ligand_id: str = "reference",
    fpocket_dir: Path | None = None,
    max_sites: int = 20,
    run_fpocket_flag: bool = True,
) -> tuple[SiteMap, dict]:
    """Build ESField site map: crystal water + fpocket hydrophobic cavity."""
    water_map = build_crystal_water_site_map(
        protein_pdb,
        ligand_path=ligand_sdf,
        protein_id=protein_id,
        ligand_id=ligand_id,
        config=CrystalWaterConfig(max_sites=max_sites),
    )
    n_water = len(water_map.sites)
    print(f"  Crystal water sites: {n_water}")

    maps = [water_map]

    # Run or use fpocket
    if run_fpocket_flag:
        fpocket_parent = Path("data/fpocket")
        fpocket_parent.mkdir(parents=True, exist_ok=True)
        fpocket_out = run_fpocket(protein_pdb, fpocket_parent)
    elif fpocket_dir:
        fpocket_out = fpocket_dir
    else:
        fpocket_out = None

    if fpocket_out and fpocket_out.exists():
        fpocket_map = parse_fpocket_site_map(
            fpocket_out,
            protein_id=protein_id,
            ligand_id=ligand_id,
            pocket_center=water_map.pocket_center,
            config=FpocketParseConfig(max_sites=max_sites),
        )
        print(f"  fpocket hydrophobic cavity sites: {len(fpocket_map.sites)}")
        maps.append(fpocket_map)

    merged = merge_site_maps(maps, merge_distance=1.0, max_sites=max_sites)
    print(f"  Merged total: {len(merged.sites)} sites")

    site_types = {}
    for s in merged.sites:
        site_types[s.site_type] = site_types.get(s.site_type, 0) + 1
    print(f"  Site types: {site_types}")

    info = {
        "protein_id": protein_id,
        "ligand_id": ligand_id,
        "pocket_center": list(merged.pocket_center),
        "n_sites": len(merged.sites),
        "site_types": site_types,
    }
    return merged, info


def prepare_pocket_for_paflow(
    protein_pdb: Path,
    output_dir: Path,
    *,
    pocket_radius: float = 10.0,
) -> Path:
    """Extract pocket PDB for PAFlow from full protein PDB.

    PAFlow's sample_for_pocket.py expects a pocket PDB (atoms within
    ~10A of ligand). This function computes the protein centroid and
    returns the full PDB path (PAFlow can work with full protein too).
    """
    atoms = read_pdb_atoms(protein_pdb)
    if not atoms:
        raise ValueError(f"no atoms in {protein_pdb}")
    pocket_center = centroid(a.coord for a in atoms)
    print(f"  Pocket center (protein centroid): ({pocket_center[0]:.1f}, {pocket_center[1]:.1f}, {pocket_center[2]:.1f})")
    return protein_pdb


def compute_pocket_volume_area(
    pocket_pdb: Path,
    ligand_sdf: Path | None = None,
) -> tuple[float, float]:
    """Compute pocket volume and area using pyKVFinder.

    If pyKVFinder fails or ligand_sdf is None, returns default estimates.
    """
    try:
        from pyKVFinder import run_workflow
        # pyKVFinder workflow
        results = run_workflow(
            str(pocket_pdb),
            step=0.6,
            probe_in=1.4,
            probe_out=4.0,
            volume_cutoff=3.0,
        )
        volume = float(results.get("volume", 0.0))
        area = float(results.get("area", 0.0))
        if volume > 0 and area > 0:
            print(f"  pyKVFinder: volume={volume:.1f}, area={area:.1f}")
            return volume, area
    except Exception as exc:
        print(f"  pyKVFinder failed: {exc}")

    # Fallback estimates based on pocket size
    print("  Using fallback volume/area estimates")
    return 250.0, 250.0


def generate_paflow_baseline(
    pocket_pdb: Path,
    output_dir: Path,
    *,
    paflow_dir: Path,
    volume: float,
    area: float,
    config: Path | None = None,
    num_samples: int = 10,
    device: str = "cuda:0",
    dry_run: bool = False,
) -> dict:
    """Run PAFlow baseline sampling for a given pocket."""
    paflow_config = config or (paflow_dir / "configs/sampling_guide.yml")
    cmd = [
        sys.executable,
        str(paflow_dir / "scripts/sample_for_pocket.py"),
        "--config", str(paflow_config),
        "--pdb_path", str(pocket_pdb),
        "--volume", str(volume),
        "--area", str(area),
        "--device", device,
        "--result_path", str(output_dir / "paflow_baseline"),
    ]
    print(f"  PAFlow command: {' '.join(cmd)}")
    if dry_run:
        print("  [DRY RUN] Not executing")
        return {"status": "dry_run", "command": " ".join(cmd)}
    scripts_dir = paflow_dir / "scripts"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(scripts_dir) + ":" + str(paflow_dir) + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd,
        cwd=str(paflow_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.stdout:
        print(result.stdout[-2000:])
    if result.stderr:
        print("STDERR:", result.stderr[-500:], file=sys.stderr)
    return {"status": "completed" if result.returncode == 0 else "failed", "returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser(description="ESField + PAFlow end-to-end pipeline")
    parser.add_argument("--protein-pdb", required=True, help="Full protein PDB with crystal waters")
    parser.add_argument("--ligand-sdf", default=None, help="Reference ligand SDF")
    parser.add_argument("--protein-id", required=True, help="Protein identifier")
    parser.add_argument("--ligand-id", default="reference")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paflow-dir", default="/root/PAFlow-main")
    parser.add_argument("--volume", type=float, default=None, help="Pocket volume (Ang^3)")
    parser.add_argument("--area", type=float, default=None, help="Pocket surface area (Ang^2)")
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--skip-fpocket", action="store_true")
    parser.add_argument("--fpocket-dir", default=None)
    parser.add_argument("--mode", choices=["prepare", "baseline", "guided", "both"], default="prepare")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_pdb = Path(args.protein_pdb)
    ligand_sdf = Path(args.ligand_sdf) if args.ligand_sdf else None

    results = {}

    # Step 1: Build site map
    print("=" * 60)
    print("Step 1: Building ESField site map")
    site_map, site_info = build_full_site_map(
        protein_pdb,
        ligand_sdf=ligand_sdf,
        protein_id=args.protein_id,
        ligand_id=args.ligand_id,
        fpocket_dir=Path(args.fpocket_dir) if args.fpocket_dir else None,
        max_sites=args.max_sites,
        run_fpocket_flag=not args.skip_fpocket,
    )
    site_map_path = output_dir / f"{args.protein_id}_site_map.json"
    site_map.write_json(site_map_path)
    print(f"  Site map saved: {site_map_path}")

    site_info_path = output_dir / f"{args.protein_id}_site_info.json"
    site_info_path.write_text(json.dumps(site_info, indent=2, ensure_ascii=False), encoding="utf-8")
    results["site_map"] = str(site_map_path)

    if args.mode == "prepare":
        print("\nSite map preparation complete. Ready for GPU generation.")
        return

    # Step 2: Prepare pocket for PAFlow
    print("\n" + "=" * 60)
    print("Step 2: Preparing pocket for PAFlow")
    pocket_pdb = prepare_pocket_for_paflow(protein_pdb, output_dir)
    results["pocket_pdb"] = str(pocket_pdb)

    # Step 3: Compute volume/area
    if args.volume is None or args.area is None:
        print("\n" + "=" * 60)
        print("Step 3: Computing pocket volume/area")
        volume, area = compute_pocket_volume_area(pocket_pdb, ligand_sdf)
    else:
        volume, area = args.volume, args.area
        print(f"\nStep 3: Using provided volume={volume}, area={area}")
    results["volume"] = volume
    results["area"] = area

    # Step 4: PAFlow baseline generation
    if args.mode in ("baseline", "both"):
        print("\n" + "=" * 60)
        print("Step 4: PAFlow baseline generation")
        baseline_result = generate_paflow_baseline(
            pocket_pdb,
            output_dir,
            paflow_dir=Path(args.paflow_dir),
            volume=volume,
            area=area,
            num_samples=args.num_samples,
            device=args.device,
            dry_run=args.dry_run,
        )
        results["baseline"] = baseline_result

    # Step 5: ESField guided generation
    if args.mode in ("guided", "both"):
        print("\n" + "=" * 60)
        print("Step 5: ESField guided generation")
        print("  NOTE: Requires PAFlow sampling loop modified with ESField injection.")
        print("  See: src/generation/adapt_paflow.py for injection point details.")
        results["guided"] = {
            "status": "not_implemented",
            "injection_target": "PAFlow models/molopt_score_model_guide.py:703",
            "esfield_adapter": "src/generation/adapt_paflow.py",
        }

    # Save results
    results_path = output_dir / f"{args.protein_id}_pipeline_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPipeline results: {results_path}")


if __name__ == "__main__":
    main()
