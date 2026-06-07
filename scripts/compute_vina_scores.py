#!/usr/bin/env python3
"""Compute Vina docking scores using obabel for PDBQT conversion.

Requires obabel (openbabel-wheel) with LD_LIBRARY_PATH set to:
  /root/miniconda3/lib/python3.12/site-packages/openbabel_wheel.libs

Usage:
    export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/openbabel_wheel.libs"
    PYTHONPATH=src python scripts/compute_vina_scores.py \
        --pocket-id 2gni --sdf-file <sdf> --protein-pdb <pdb> --output-csv <csv>
"""

import argparse, csv, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from rdkit import Chem

ESFIELD_ROOT = str(Path(__file__).resolve().parents[1])

OBO_LIB = "/root/miniconda3/lib/python3.12/site-packages/openbabel_wheel.libs"
CONDA_BIN = "/root/miniconda3/bin"
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = OBO_LIB + ":" + ENV.get("LD_LIBRARY_PATH", "")
ENV["PATH"] = CONDA_BIN + ":" + ENV.get("PATH", "")
OBO_CMD = os.path.join(CONDA_BIN, "obabel")
VINA_CMD = "/usr/bin/vina"


def sdf_to_pdbqt(sdf_content, output_path):
    """Convert SDF string to PDBQT using obabel."""
    tmp_sdf = output_path.replace(".pdbqt", "_tmp.sdf")
    with open(tmp_sdf, "w") as f:
        f.write(sdf_content)
    cmd = [OBO_CMD, tmp_sdf, "-O", output_path, "--gen3d"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=ENV)
    if os.path.exists(tmp_sdf):
        os.unlink(tmp_sdf)
    return r.returncode == 0 and os.path.exists(output_path)


def protein_to_pdbqt(protein_pdb, output_path):
    """Convert protein PDB to PDBQT using obabel."""
    if os.path.exists(output_path):
        return True
    cmd = [OBO_CMD, protein_pdb, "-O", output_path, "-xr"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=ENV)
    return r.returncode == 0 and os.path.exists(output_path)


def get_combined_box(pocket_pdb, mol):
    """Compute box covering both pocket and ligand centroid."""
    # Read pocket atoms
    pocket_coords = []
    try:
        with open(pocket_pdb) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        pocket_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except ValueError:
                        continue
    except Exception:
        pocket_coords = []

    # Ligand coords
    conf = mol.GetConformer()
    lig_coords = np.array([[conf.GetAtomPosition(a.GetIdx()).x,
                            conf.GetAtomPosition(a.GetIdx()).y,
                            conf.GetAtomPosition(a.GetIdx()).z]
                           for a in mol.GetAtoms() if a.GetAtomicNum() > 0])

    if len(pocket_coords) > 0:
        pocket_coords = np.array(pocket_coords)
        p_center = pocket_coords.mean(axis=0)
        p_span = pocket_coords.max(axis=0) - pocket_coords.min(axis=0)
    else:
        p_center = np.zeros(3)
        p_span = np.zeros(3)

    l_center = lig_coords.mean(axis=0)
    l_span = lig_coords.max(axis=0) - lig_coords.min(axis=0)

    # Center between pocket and ligand, size covers both + padding
    center = ((p_center + l_center) / 2).tolist()
    combined_span = np.maximum(p_span, l_span) + 10  # 10Å padding
    size = [float(max(22, min(35, s))) for s in combined_span]
    return center, size


def score_one(sdf_content, mol_idx, protein_pdbqt, pocket_pdb, work_dir, seed):
    """Score one molecule: SDF → obabel PDBQT → vina --score_only."""
    lig_path = os.path.join(work_dir, f"lig_{mol_idx}.pdbqt")
    sdf_path = os.path.join(work_dir, f"mol_{mol_idx}.sdf")

    # Write SDF to temp file
    with open(sdf_path, "w") as f:
        f.write(sdf_content)

    # Convert to PDBQT using obabel
    if not sdf_to_pdbqt(sdf_content, lig_path):
        return {"mol_index": mol_idx, "vina_score": None, "success": False, "error": "obabel conversion failed"}

    # Read molecule to compute box
    mol = Chem.SDMolSupplier(sdf_path, sanitize=False)[0]
    if mol is None:
        return {"mol_index": mol_idx, "vina_score": None, "success": False, "error": "RDKit read failed"}
    center, size = get_combined_box(pocket_pdb, mol)

    out_path = lig_path.replace(".pdbqt", "_out.pdbqt")
    cmd = [
        VINA_CMD, "--receptor", protein_pdbqt, "--ligand", lig_path, "--out", out_path,
        "--center_x", f"{center[0]:.2f}", "--center_y", f"{center[1]:.2f}", "--center_z", f"{center[2]:.2f}",
        "--size_x", f"{size[0]:.1f}", "--size_y", f"{size[1]:.1f}", "--size_z", f"{size[2]:.1f}",
        "--score_only", "--seed", str(seed), "--num_modes", "1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=ENV)
        stdout = r.stdout + "\n" + r.stderr

        # Parse: "Estimated Free Energy of Binding   : -8.3 (kcal/mol)"
        score = None
        for line in stdout.split("\n"):
            if "Estimated Free Energy of Binding" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    val = parts[1].strip().split()[0]
                    try:
                        score = float(val)
                    except ValueError:
                        pass
                break

        # Cleanup
        for p in [lig_path, out_path, sdf_path]:
            if os.path.exists(p):
                os.unlink(p)

        return {"mol_index": mol_idx, "vina_score": score, "success": score is not None,
                "error": None if score is not None else "Could not parse Vina output"}
    except Exception as e:
        return {"mol_index": mol_idx, "vina_score": None, "success": False, "error": str(e)[:100]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-id", required=True)
    parser.add_argument("--sdf-file", required=True)
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--pocket-pdb", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mols", type=int, default=0, help="Max molecules (0=all)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix=f"vina_{args.pocket_id}_")

    print(f"[1/3] Loading SDF + preparing protein")
    mols = list(Chem.SDMolSupplier(args.sdf_file, sanitize=False))
    mols = [m for m in mols if m is not None]
    if args.max_mols > 0:
        mols = mols[:args.max_mols]
    print(f"  {len(mols)} molecules")

    prot_pdbqt = os.path.join(os.path.dirname(args.output_csv) or ".",
                              f"{args.pocket_id}_protein.pdbqt")
    if not protein_to_pdbqt(args.protein_pdb, prot_pdbqt):
        print(f"  ERROR: Protein PDBQT preparation failed")
        sys.exit(1)
    print(f"  Protein PDBQT: {prot_pdbqt}")

    print(f"[2/3] Scoring {len(mols)} molecules ({args.n_workers} workers)...")
    t0 = time.time()

    # Export SDF content per molecule
    sdf_contents = []
    for i, mol in enumerate(mols):
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            pass
        tmp_sdf = os.path.join(work_dir, f"_tmp_{i}.sdf")
        try:
            writer = Chem.SDWriter(tmp_sdf)
            writer.write(mol)
            writer.close()
            with open(tmp_sdf) as f:
                sdf_contents.append(f.read())
        except Exception as e:
            sdf_contents.append("")  # placeholder for failed molecule

    pocket_pdb = args.pocket_pdb or args.protein_pdb
    tasks = [(sdf_contents[i], i, prot_pdbqt, pocket_pdb, work_dir, args.seed + i)
             for i in range(len(mols)) if sdf_contents[i]]
    results = [{"mol_index": i, "vina_score": None, "success": False, "error": "SDF export failed"}
               for i in range(len(mols)) if not sdf_contents[i]]
    with Pool(processes=args.n_workers) as pool:
        for result in pool.starmap(score_one, tasks, chunksize=3):
            results.append(result)
            if len(results) % max(1, len(mols) // 5) == 0:
                n_ok = sum(1 for r in results if r["success"])
                print(f"  {len(results)}/{len(mols)}, {n_ok} OK")

    n_ok = sum(1 for r in results if r["success"])
    elapsed = time.time() - t0
    print(f"  Done: {n_ok}/{len(mols)} successful ({elapsed:.1f}s, {elapsed/max(1,len(mols)):.1f}s/mol)")

    print(f"[3/3] Saving to {args.output_csv}")
    results.sort(key=lambda r: r["mol_index"])
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_index", "vina_score", "success", "error"])
        w.writeheader()
        for r in results:
            w.writerow({k: (v if v is not None else "") for k, v in r.items()})

    scores = [r["vina_score"] for r in results if r["success"] and r["vina_score"] is not None]
    if scores:
        print(f"  Vina: mean={np.mean(scores):.1f}±{np.std(scores):.1f}, min={np.min(scores):.1f}, max={np.max(scores):.1f} kcal/mol")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
