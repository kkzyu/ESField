"""Export PyMOL scripts for visual inspection of ESField sites."""

from __future__ import annotations

import argparse
from pathlib import Path

from site_detection.site_schema import SiteMap

SITE_COLORS = {
    "high_energy_water": "red",
    "stable_water": "blue",
    "hydrophobic_cavity": "yellow",
}


def write_pymol_script(
    site_map: SiteMap,
    *,
    output: str | Path,
    receptor: str | Path | None = None,
    ligand: str | Path | None = None,
) -> None:
    lines: list[str] = ["reinitialize", "bg_color white"]
    if receptor:
        lines.append(f"load {Path(receptor).as_posix()}, receptor")
        lines.append("show cartoon, receptor")
        lines.append("color gray70, receptor")
    if ligand:
        lines.append(f"load {Path(ligand).as_posix()}, ligand")
        lines.append("show sticks, ligand")
        lines.append("color green, ligand")
    for site in site_map.sites:
        name = f"site_{site.site_id}_{site.site_type}"
        color = SITE_COLORS.get(site.site_type, "white")
        x, y, z = site.center
        lines.append(f"pseudoatom {name}, pos=[{x:.3f}, {y:.3f}, {z:.3f}], vdw={site.radius:.3f}")
        lines.append(f"show spheres, {name}")
        lines.append(f"set sphere_transparency, 0.45, {name}")
        lines.append(f"color {color}, {name}")
    lines.append("zoom")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a PyMOL script for an ESField site map.")
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receptor", default=None)
    parser.add_argument("--ligand", default=None)
    args = parser.parse_args()
    write_pymol_script(SiteMap.read_json(args.site_map), output=args.output, receptor=args.receptor, ligand=args.ligand)
    print(f"wrote PyMOL script: {args.output}")


if __name__ == "__main__":
    main()

