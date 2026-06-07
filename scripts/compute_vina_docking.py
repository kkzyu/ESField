#!/usr/bin/env python3
"""Full Vina docking with MMFF94 pre-minimization (Phase IIb).

Pipeline:  MMFF94 minimize → obabel PDBQT → Vina full docking (exhaustiveness=8)

Key difference from Phase II: uses FULL conformational search (not --score_only)
on pre-minimized molecules, yielding Vina scores in the -10 to +10 kcal/mol range
instead of 100-300 for raw unminimized poses.

Usage:
    PYTHONPATH=src python scripts/compute_vina_docking.py \
        --pocket-id 2gni --sdf-file <sdf> --protein-pdb <pdb> \
        --pocket-pdb <pocket.pdb> --output-csv <csv> --mode full_dock
"""

import argparse, csv, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from rdkit import Chem

ESFIELD_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ESFIELD_ROOT, "src"))

from utils.minimize_molecule import minimize_molecule_with_mmff

# Paths
OBO_LIB = "/root/miniconda3/lib/python3.12/site-packages/openbabel_wheel.libs"
CONDA_BIN = "/root/miniconda3/bin"
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = OBO_LIB + ":" + ENV.get("LD_LIBRARY_PATH", "")
ENV["PATH"] = CONDA_BIN + ":" + ENV.get("PATH", "")
OBO_CMD = os.path.join(CONDA_BIN, "obabel")
VINA_CMD = "/usr/bin/vina"


def get_docking_box(pocket_pdb, padding=8.0):
    """Compute Vina box from pocket residue coordinates."""
    coords = []
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    continue
    if not coords:
        return [0, 0, 0], [20, 20, 20]
    coords = np.array(coords)
    center = coords.mean(axis=0).tolist()
    span = (coords.max(axis=0) - coords.min(axis=0)) + padding
    size = [float(max(20, min(30, s))) for s in span]
    return center, size


def mol_to_pdbqt_via_obabel(mol, output_path, work_dir, mol_idx):
    """Convert RDKit mol to PDBQT via obabel (SDF intermediate)."""
    sdf_path = os.path.join(work_dir, f"_sdf_{mol_idx}.sdf")
    try:
        w = Chem.SDWriter(sdf_path)
        w.write(mol)
        w.close()
    except Exception:
        return False

    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) < 50:
        return False

    cmd = [OBO_CMD, sdf_path, "-O", output_path, "--gen3d"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=ENV)
    return r.returncode == 0 and os.path.exists(output_path)


def dock_one_molecule(args):
    """Full pipeline: minimize → obabel → Vina dock. Called by Pool."""
    mol, mol_idx, protein_pdbqt, box_center, box_size, work_dir, seed, mode = args

    result = {
        "mol_index": mol_idx,
        "vina_score": None, "vina_score_raw": None,
        "minimized": False, "converged": False,
        "ff_name": "none", "energy_before": None, "energy_after": None,
        "success": False, "error": None,
    }

    if mol is None:
        result["error"] = "mol is None"
        return result

    try:
        # Step 1: MMFF94 minimization
        min_mol, info = minimize_molecule_with_mmff(mol, max_iters=200, force_tolerance=0.01)
        result["minimized"] = True
        result["converged"] = info["converged"]
        result["ff_name"] = info["ff_name"]
        result["energy_before"] = info.get("energy_before")
        result["energy_after"] = info.get("energy_after")

        if min_mol is None:
            result["error"] = "minimization returned None"
            return result

        # Step 2: Convert to PDBQT
        lig_path = os.path.join(work_dir, f"lig_{mol_idx}.pdbqt")
        if not mol_to_pdbqt_via_obabel(min_mol, lig_path, work_dir, mol_idx):
            result["error"] = "obabel conversion failed"
            return result

        # Step 3: Vina docking
        out_path = lig_path.replace(".pdbqt", "_out.pdbqt")
        if mode == "full_dock":
            # Full conformational search
            cmd = [
                VINA_CMD, "--receptor", protein_pdbqt, "--ligand", lig_path,
                "--out", out_path,
                "--center_x", f"{box_center[0]:.2f}",
                "--center_y", f"{box_center[1]:.2f}",
                "--center_z", f"{box_center[2]:.2f}",
                "--size_x", f"{box_size[0]:.1f}",
                "--size_y", f"{box_size[1]:.1f}",
                "--size_z", f"{box_size[2]:.1f}",
                "--exhaustiveness", "8",
                "--num_modes", "9",
                "--seed", str(seed),
            ]
        else:
            # --score_only (original mode, for comparison)
            cmd = [
                VINA_CMD, "--receptor", protein_pdbqt, "--ligand", lig_path,
                "--out", out_path,
                "--center_x", f"{box_center[0]:.2f}",
                "--center_y", f"{box_center[1]:.2f}",
                "--center_z", f"{box_center[2]:.2f}",
                "--size_x", f"{box_size[0]:.1f}",
                "--size_y", f"{box_size[1]:.1f}",
                "--size_z", f"{box_size[2]:.1f}",
                "--score_only", "--seed", str(seed), "--num_modes", "1",
            ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=ENV)
        stdout = r.stdout + "\n" + r.stderr

        # Parse: first mode line gives best score
        # "1    -8.3    0.000    0.000"
        for line in stdout.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    score = float(parts[1])
                    if -100 < score < 500:
                        result["vina_score"] = score
                        break
                except ValueError:
                    continue

        # Also try "Estimated Free Energy of Binding"
        if result["vina_score"] is None:
            for line in stdout.split("\n"):
                if "Estimated Free Energy of Binding" in line:
                    try:
                        val = line.split(":")[1].strip().split()[0]
                        result["vina_score"] = float(val)
                    except Exception:
                        pass

        # Cleanup
        for p in [lig_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)

        result["success"] = result["vina_score"] is not None
        if not result["success"]:
            result["error"] = "Could not parse Vina score"

    except Exception as e:
        result["error"] = str(e)[:150]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-id", required=True)
    parser.add_argument("--sdf-file", required=True)
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--pocket-pdb", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--mode", default="full_dock", choices=["full_dock", "score_only"])
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mols", type=int, default=0)
    parser.add_argument("--pre-minimized", action="store_true",
                        help="SDF already minimized, skip MMFF step")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix=f"dock_{args.pocket_id}_")

    # Load molecules
    print(f"[1/4] Loading {args.sdf_file}")
    mols = list(Chem.SDMolSupplier(args.sdf_file, sanitize=False))
    mols = [m for m in mols if m is not None]
    if args.max_mols > 0:
        mols = mols[:args.max_mols]
    print(f"  {len(mols)} molecules")

    # Protein PDBQT
    print(f"[2/4] Preparing protein PDBQT")
    prot_pdbqt = os.path.join(os.path.dirname(args.output_csv),
                              f"{args.pocket_id}_protein.pdbqt")
    if not os.path.exists(prot_pdbqt):
        cmd = [OBO_CMD, args.protein_pdb, "-O", prot_pdbqt, "-xr"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=ENV)
        if r.returncode != 0:
            print(f"  ERROR: Protein PDBQT failed: {r.stderr[:200]}")
            sys.exit(1)
    print(f"  {prot_pdbqt}")

    # Docking box
    pocket_pdb = args.pocket_pdb or args.protein_pdb
    box_center, box_size = get_docking_box(pocket_pdb)
    print(f"  Box: center={[f'{c:.1f}' for c in box_center]}, "
          f"size={[f'{s:.1f}' for s in box_size]}")

    # Dock all molecules
    print(f"[3/4] Docking {len(mols)} molecules "
          f"(mode={args.mode}, pre_minimized={args.pre_minimized}, {args.n_workers} workers)...")
    t0 = time.time()

    # If pre-minimized, pass through; otherwise minimize in dock_one_molecule
    tasks = [(m, i, prot_pdbqt, box_center, box_size, work_dir, args.seed + i, args.mode)
             for i, m in enumerate(mols)]

    results = []
    with Pool(processes=args.n_workers) as pool:
        for result in pool.imap_unordered(dock_one_molecule, tasks, chunksize=2):
            results.append(result)
            if len(results) % max(1, len(mols) // 5) == 0:
                n_ok = sum(1 for r in results if r["success"])
                print(f"  {len(results)}/{len(mols)}, {n_ok} OK")

    n_ok = sum(1 for r in results if r["success"])
    elapsed = time.time() - t0
    print(f"  Done: {n_ok}/{len(mols)} successful ({elapsed:.1f}s, "
          f"{elapsed/max(1,len(mols)):.1f}s/mol)")

    # Save CSV
    print(f"[4/4] Saving to {args.output_csv}")
    results.sort(key=lambda r: r["mol_index"])
    fields = ["mol_index", "vina_score", "minimized", "converged",
              "ff_name", "energy_before", "energy_after", "success", "error"]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: ("" if v is None else v) for k, v in r.items()}
            w.writerow(row)

    scores = [r["vina_score"] for r in results if r["success"] and r["vina_score"] is not None]
    if scores:
        print(f"  Vina: mean={np.mean(scores):.1f}±{np.std(scores):.1f}, "
              f"min={np.min(scores):.1f}, max={np.max(scores):.1f} kcal/mol")
    n_conv = sum(1 for r in results if r.get("converged"))
    print(f"  Minimization converged: {n_conv}/{len(mols)}")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
