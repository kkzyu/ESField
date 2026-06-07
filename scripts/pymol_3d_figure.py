#!/usr/bin/env python3
"""PyMOL Publication-Quality 3D Molecular Comparison Figure (P0-2).

Creates Figure 3 from the ESField paper: side-by-side comparison of
  (a) Hard-fix worst Vina molecule — steric clashes highlighted
  (b) Kinematic best Vina molecule — relaxed, complementary pose
  (c) Crystal reference ligand (if available)

HEW sites are shown as semi-transparent blue spheres.

Usage:
  # Inside PyMOL:  run pymol_3d_figure.py
  # CLI (headless): pymol -cq pymol_3d_figure.py -- --pocket 3mfw

Requirements:
  - PyMOL 2.5+ (open-source or licensed)
  - Protein pocket PDB file
  - Generated molecule SDF files
  - Reference ligand SDF file (optional)
"""

import os
import sys
import argparse
from pathlib import Path

# ── Configuration ──
POCKET_PATHS = {
    "3mfw": {
        "protein": "/root/autodl-tmp/data/PDB/P-L/2001-2010/3mfw/3mfw_pocket.pdb",
        "ref_ligand": "/root/autodl-tmp/data/PDB/P-L/2001-2010/3mfw/3mfw_ligand.sdf",
    },
    "6o4x": {
        "protein": "/root/autodl-tmp/data/PDB/P-L/2011-2019/6o4x/6o4x_pocket.pdb",
        "ref_ligand": "/root/autodl-tmp/data/PDB/P-L/2011-2019/6o4x/6o4x_ligand.sdf",
    },
}

# HEW site centers from site maps (pre-computed)
HEW_SITES = {
    "3mfw": [
        (4.586, 17.821, 25.343),
        (4.456, 19.371, 27.695),
        (4.374, 22.142, 26.882),
        (2.428, 25.995, 26.113),
        (10.417, 23.109, 31.419),
        (4.118, 27.286, 21.701),
        (8.337, 17.340, 27.854),
    ],
    "6o4x": [
        (101.897, 53.355, -10.645),
        (97.064, 52.719, -5.684),
        (96.185, 49.570, -10.403),
        (99.551, 52.602, -9.122),
        (99.219, 49.814, -5.538),
        (99.693, 53.305, -6.536),
    ],
}

# PyMOL rendering settings for publication quality
RENDER_SETTINGS = """
# ── Background ──
set bg_color, black

# ── Ray tracing ──
set ray_shadows, 1
set ray_trace_mode, 1
set ray_trace_fog, 0
set antialias, 2
set ambient, 0.2
set direct, 0.6
set reflect, 0.4

# ── General display ──
set depth_cue, 1
set specular, 1
set spec_power, 200
set stick_radius, 0.15
set sphere_scale, 0.3
set cartoon_side_chain_helper, 1

# ── Surface ──
set surface_quality, 2
set transparency_mode, 2

# ── Labels ──
set label_size, 24
set label_font_id, 5
set label_color, white
"""


def setup_scene(pocket_id, hardfix_sdf, kinematic_sdf, ref_sdf=None,
                output_prefix="fig3"):
    """Create the 3D comparison scene in PyMOL.

    Call this function from within a PyMOL session.
    """
    try:
        import pymol
        from pymol import cmd, stored
    except ImportError:
        print("ERROR: This script must be run inside PyMOL.")
        print("Usage: pymol -cq pymol_3d_figure.py -- --pocket 3mfw")
        sys.exit(1)

    cfg = POCKET_PATHS.get(pocket_id, {})
    protein_pdb = cfg.get("protein", "")
    if not ref_sdf:
        ref_sdf = cfg.get("ref_ligand", "")

    hew_centers = HEW_SITES.get(pocket_id, [])

    # ── Reinitialize ──
    cmd.reinitialize()
    cmd.set("bg_color", "black")

    # ── Load protein pocket ──
    if os.path.exists(protein_pdb):
        cmd.load(protein_pdb, "protein")
        cmd.hide("everything", "protein")
        cmd.show("surface", "protein")
        cmd.set("surface_color", "grey70", "protein")
        cmd.set("transparency", 0.3, "protein")
    else:
        print(f"WARNING: Protein PDB not found: {protein_pdb}")

    # ── Create HEW site spheres ──
    for i, (x, y, z) in enumerate(hew_centers):
        obj_name = f"HEW_{i}"
        cmd.pseudoatom(obj_name, pos=(x, y, z), vdw=1.4, color="blue")
        cmd.set("sphere_transparency", 0.45, obj_name)
        # Label
        cmd.label(f"{obj_name} and name PS1", f"HEW-{i+1}")

    # ── Load and style molecules ──
    # Panel (a): Hard-fix worst molecule
    if hardfix_sdf and os.path.exists(hardfix_sdf):
        cmd.load(hardfix_sdf, "hardfix_worst")
        cmd.hide("everything", "hardfix_worst")
        cmd.show("sticks", "hardfix_worst")
        cmd.color("red", "hardfix_worst")
        cmd.set("stick_radius", 0.2, "hardfix_worst")
        # Show clash spheres at anchors
        cmd.show("spheres", "hardfix_worst and name C*")
        cmd.set("sphere_scale", 0.4, "hardfix_worst")
        cmd.set("sphere_transparency", 0.6, "hardfix_worst")
    else:
        print(f"WARNING: Hard-fix SDF not found: {hardfix_sdf}")

    # Panel (b): Kinematic best molecule
    if kinematic_sdf and os.path.exists(kinematic_sdf):
        cmd.load(kinematic_sdf, "kinematic_best")
        cmd.hide("everything", "kinematic_best")
        cmd.show("sticks", "kinematic_best")
        cmd.color("green", "kinematic_best")
        cmd.set("stick_radius", 0.2, "kinematic_best")
    else:
        print(f"WARNING: Kinematic SDF not found: {kinematic_sdf}")

    # Panel (c): Crystal reference ligand
    if ref_sdf and os.path.exists(ref_sdf):
        cmd.load(ref_sdf, "crystal_ligand")
        cmd.hide("everything", "crystal_ligand")
        cmd.show("sticks", "crystal_ligand")
        cmd.color("yellow", "crystal_ligand")
        cmd.set("stick_radius", 0.2, "crystal_ligand")
    else:
        print(f"WARNING: Reference ligand SDF not found: {ref_sdf}")

    # ── Apply rendering settings ──
    cmd.do(RENDER_SETTINGS)

    # ── Zoom to pocket ──
    cmd.zoom("protein", 5.0)
    cmd.orient("protein")

    # ── Create views for each panel ──
    # We'll render the same scene with different object visibility
    # for a combined figure

    output_dir = Path(output_prefix).parent or Path(".")
    os.makedirs(output_dir, exist_ok=True)

    # View 1: Hard-fix (red) + protein + HEW
    cmd.hide("everything", "kinematic_best")
    cmd.hide("everything", "crystal_ligand")
    cmd.show("sticks", "hardfix_worst")
    cmd.show("spheres", "hardfix_worst and name C*")
    cmd.ray(2400, 1800)
    cmd.png(str(output_dir / f"{output_prefix}_panel_a_hardfix.png"), dpi=300)
    print(f"Saved panel (a): hard-fix")

    # View 2: Kinematic (green) + protein + HEW
    cmd.hide("everything", "hardfix_worst")
    cmd.show("sticks", "kinematic_best")
    cmd.ray(2400, 1800)
    cmd.png(str(output_dir / f"{output_prefix}_panel_b_kinematic.png"), dpi=300)
    print(f"Saved panel (b): kinematic")

    # View 3: Crystal ligand (yellow) + protein + HEW
    cmd.hide("everything", "kinematic_best")
    cmd.show("sticks", "crystal_ligand")
    cmd.ray(2400, 1800)
    cmd.png(str(output_dir / f"{output_prefix}_panel_c_crystal.png"), dpi=300)
    print(f"Saved panel (c): crystal ligand")

    # Combined view: all three visible
    cmd.show("sticks", "hardfix_worst")
    cmd.show("spheres", "hardfix_worst and name C*")
    cmd.show("sticks", "kinematic_best")
    cmd.show("sticks", "crystal_ligand")
    cmd.ray(2400, 1800)
    cmd.png(str(output_dir / f"{output_prefix}_combined.png"), dpi=300)
    print(f"Saved combined view")

    # ── Save PyMOL session ──
    cmd.save(str(output_dir / f"{output_prefix}.pse"))
    print(f"PyMOL session saved")

    print(f"\nAll outputs saved to {output_dir}/")


def create_cli_script():
    """Create a self-contained PyMOL script that works without arguments."""
    return r'''
# PyMOL Figure 3 Generation Script
# Run: pymol -cq this_script.py

import sys, os
sys.path.insert(0, '/root/ESField/scripts')
from pymol_3d_figure import setup_scene

# Configure which pocket and molecules to visualize
POCKET = "3mfw"
HARDFIX_SDF = "/root/ESField/experiments/targetdiff_native_guided/3mfw/hard_fix/sdf/hard_fix_000.sdf"
KINEMATIC_SDF = "/root/ESField/experiments/targetdiff_native_guided/3mfw/kinematic/sdf/kinematic_000.sdf"
REF_SDF = "/root/autodl-tmp/data/PDB/P-L/2001-2010/3mfw/3mfw_ligand.sdf"
OUTPUT = "/root/ESField/paper_latex/figures/fig3_3D_comparison"

# Override with CLI args if provided
if '--' in sys.argv:
    idx = sys.argv.index('--')
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pocket', default=POCKET)
    p.add_argument('--hardfix-sdf', default=HARDFIX_SDF)
    p.add_argument('--kinematic-sdf', default=KINEMATIC_SDF)
    p.add_argument('--ref-sdf', default=REF_SDF)
    p.add_argument('--output', default=OUTPUT)
    args = p.parse_args(sys.argv[idx+1:])
    POCKET = args.pocket
    HARDFIX_SDF = args.hardfix_sdf
    KINEMATIC_SDF = args.kinematic_sdf
    REF_SDF = args.ref_sdf
    OUTPUT = args.output

# Run
setup_scene(POCKET, HARDFIX_SDF, KINEMATIC_SDF, REF_SDF, OUTPUT)
'''


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality 3D molecular comparison figure"
    )
    parser.add_argument("--pocket", default="3mfw", choices=["3mfw", "6o4x"])
    parser.add_argument("--hardfix-sdf", default=None,
                        help="SDF file of worst hard-fix molecule")
    parser.add_argument("--kinematic-sdf", default=None,
                        help="SDF file of best kinematic molecule")
    parser.add_argument("--ref-sdf", default=None,
                        help="SDF file of crystal reference ligand")
    parser.add_argument("--output", default="paper_latex/figures/fig3_3D_comparison",
                        help="Output prefix for PNG files")
    parser.add_argument("--write-pymol-script", action="store_true",
                        help="Write a standalone PyMOL script and exit")
    args = parser.parse_args()

    if args.write_pymol_script:
        script_path = "/root/ESField/scripts/run_pymol_figure.py"
        with open(script_path, "w") as f:
            f.write(create_cli_script())
        print(f"PyMOL script written to {script_path}")
        print(f"Run: pymol -cq {script_path} -- --pocket 3mfw")
        return

    # Set defaults if not provided
    pocket = args.pocket
    hardfix_sdf = args.hardfix_sdf
    kinematic_sdf = args.kinematic_sdf
    ref_sdf = args.ref_sdf or POCKET_PATHS[pocket]["ref_ligand"]

    print("=" * 60)
    print("PyMOL 3D Comparison Figure Generator")
    print("=" * 60)
    print(f"Pocket: {pocket}")
    print(f"Hard-fix SDF: {hardfix_sdf or 'NOT SET'}")
    print(f"Kinematic SDF: {kinematic_sdf or 'NOT SET'}")
    print(f"Ref ligand: {ref_sdf or 'NOT SET'}")
    print()

    # If running inside PyMOL, go ahead
    try:
        import pymol
        setup_scene(pocket, hardfix_sdf, kinematic_sdf, ref_sdf, args.output)
    except ImportError:
        print("PyMOL not available. Writing standalone script instead.")
        script_path = "/root/ESField/scripts/run_pymol_figure.py"
        with open(script_path, "w") as f:
            f.write(create_cli_script())
        print(f"\nPyMOL script written to {script_path}")
        print("After installing PyMOL, run:")
        print(f"  pymol -cq {script_path} -- --pocket {pocket}")
        print("\nOr provide SDF paths:")
        print(f"  pymol -cq {script_path} -- --pocket {pocket} \\")
        print(f"    --hardfix-sdf <path> --kinematic-sdf <path>")


if __name__ == "__main__":
    main()
