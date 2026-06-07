#!/usr/bin/env python3
"""PyMOL Figure 3 — 3D Molecular Comparison (Portable, Local-Use Ready)

Copy this script + data files to your local machine, then:
  pymol -cq run_pymol_figure.py -- \
    --protein ./3mfw_pocket.pdb \
    --hardfix ./hardfix_worst.sdf \
    --kinematic ./kinematic_best.sdf \
    --ref-ligand ./3mfw_ligand.sdf \
    --output ./fig3_output

Files needed (copy from server to local):
  - <pocket>_pocket.pdb   (protein pocket structure)
  - <pocket>_ligand.sdf   (crystal reference ligand, optional)
  - One worst-Vina hard-fix molecule SDF
  - One best-Vina kinematic molecule SDF

Server paths for reference:
  /root/autodl-tmp/data/PDB/P-L/<year>/<pocket>/<pocket>_pocket.pdb
  /root/autodl-tmp/data/PDB/P-L/<year>/<pocket>/<pocket>_ligand.sdf
  /root/ESField/experiments/targetdiff_native_guided/<pocket>/<mode>/sdf/
"""

import argparse
import os
import sys

# ── HEW site coordinates (pre-computed) ──
# These define the blue sphere positions in the figure
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

RENDER_SETTINGS = """
set bg_color, black
set ray_shadows, 1
set ray_trace_mode, 1
set antialias, 2
set ambient, 0.2
set direct, 0.6
set reflect, 0.4
set depth_cue, 1
set specular, 1
set stick_radius, 0.15
set sphere_scale, 0.3
set surface_quality, 2
set transparency_mode, 2
set label_size, 24
set label_font_id, 5
set label_color, white
"""


def setup_scene(protein_pdb, hardfix_sdf, kinematic_sdf, ref_sdf,
                hew_sites, output_prefix):
    """Create the publication-quality 3D comparison figure."""
    from pymol import cmd

    cmd.reinitialize()
    cmd.set("bg_color", "black")

    # ── Protein surface (grey, semi-transparent) ──
    if os.path.exists(protein_pdb):
        cmd.load(protein_pdb, "protein")
        cmd.hide("everything", "protein")
        cmd.show("surface", "protein")
        cmd.set("surface_color", "grey70", "protein")
        cmd.set("transparency", 0.3, "protein")
        print(f"  Loaded protein: {protein_pdb}")
    else:
        print(f"  WARNING: protein not found: {protein_pdb}")

    # ── HEW site spheres (blue, semi-transparent) ──
    for i, (x, y, z) in enumerate(hew_sites):
        name = f"HEW_{i}"
        cmd.pseudoatom(name, pos=(x, y, z), vdw=1.4, color="blue")
        cmd.set("sphere_transparency", 0.45, name)
        cmd.label(f"{name} and name PS1", f"HEW-{i+1}")

    # ── Hard-fix worst molecule (red sticks + clash spheres) ──
    if hardfix_sdf and os.path.exists(hardfix_sdf):
        cmd.load(hardfix_sdf, "hardfix")
        cmd.hide("everything", "hardfix")
        cmd.show("sticks", "hardfix")
        cmd.color("red", "hardfix")
        cmd.set("stick_radius", 0.2, "hardfix")
        cmd.show("spheres", "hardfix and name C*")
        cmd.set("sphere_scale", 0.4, "hardfix")
        cmd.set("sphere_transparency", 0.6, "hardfix")
        print(f"  Loaded hard-fix: {hardfix_sdf}")

    # ── Kinematic best molecule (green sticks) ──
    if kinematic_sdf and os.path.exists(kinematic_sdf):
        cmd.load(kinematic_sdf, "kinematic")
        cmd.hide("everything", "kinematic")
        cmd.show("sticks", "kinematic")
        cmd.color("green", "kinematic")
        cmd.set("stick_radius", 0.2, "kinematic")
        print(f"  Loaded kinematic: {kinematic_sdf}")

    # ── Crystal reference ligand (yellow sticks) ──
    if ref_sdf and os.path.exists(ref_sdf):
        cmd.load(ref_sdf, "crystal")
        cmd.hide("everything", "crystal")
        cmd.show("sticks", "crystal")
        cmd.color("yellow", "crystal")
        cmd.set("stick_radius", 0.2, "crystal")
        print(f"  Loaded ref ligand: {ref_sdf}")

    # ── Rendering ──
    cmd.do(RENDER_SETTINGS)
    cmd.zoom("protein", 5.0)

    os.makedirs(output_prefix if os.path.dirname(output_prefix) else ".", exist_ok=True)

    # Panel (a): Hard-fix only
    cmd.hide("everything", "kinematic")
    cmd.hide("everything", "crystal")
    cmd.show("sticks", "hardfix")
    cmd.show("spheres", "hardfix and name C*")
    cmd.ray(2400, 1800)
    cmd.png(f"{output_prefix}_a_hardfix.png", dpi=300)
    print(f"  Saved panel (a): hard-fix")

    # Panel (b): Kinematic only
    cmd.hide("everything", "hardfix")
    cmd.show("sticks", "kinematic")
    cmd.ray(2400, 1800)
    cmd.png(f"{output_prefix}_b_kinematic.png", dpi=300)
    print(f"  Saved panel (b): kinematic")

    # Panel (c): Crystal reference
    cmd.hide("everything", "kinematic")
    cmd.show("sticks", "crystal")
    cmd.ray(2400, 1800)
    cmd.png(f"{output_prefix}_c_crystal.png", dpi=300)
    print(f"  Saved panel (c): crystal")

    # Combined
    cmd.show("sticks", "hardfix")
    cmd.show("spheres", "hardfix and name C*")
    cmd.show("sticks", "kinematic")
    cmd.show("sticks", "crystal")
    cmd.ray(2400, 1800)
    cmd.png(f"{output_prefix}_combined.png", dpi=300)
    print(f"  Saved combined view")

    cmd.save(f"{output_prefix}.pse")
    print(f"  Session saved: {output_prefix}.pse")
    print(f"\nDone! All outputs at {output_prefix}_*")


if __name__ == "__main__":
    # When run directly (pymol -cq this_script.py), use hardcoded paths
    # that the user should edit. Or pass via -- args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", default="./pocket.pdb")
    parser.add_argument("--hardfix", default="./hardfix.sdf")
    parser.add_argument("--kinematic", default="./kinematic.sdf")
    parser.add_argument("--ref-ligand", default="./ref_ligand.sdf")
    parser.add_argument("--pocket-id", default="3mfw", choices=["3mfw", "6o4x"])
    parser.add_argument("--output", default="./fig3_output")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    hew = HEW_SITES.get(args.pocket_id, HEW_SITES["3mfw"])

    print("=" * 50)
    print("ESField Figure 3: 3D Molecular Comparison")
    print("=" * 50)

    setup_scene(args.protein, args.hardfix, args.kinematic,
                args.ref_ligand, hew, args.output)
