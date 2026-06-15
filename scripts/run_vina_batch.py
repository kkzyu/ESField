#!/usr/bin/env python3
"""Batch Vina docking for DrugFlow main experiment molecules.

CPU-only — designed to run alongside GPU TargetDiff generation.
Uses meeko for SDF→PDBQT conversion, AutoDock Vina 1.2.3 for docking.

Usage:
  python scripts/run_vina_batch.py --pocket 3mfw \
    --base-dir /root/ESField/experiments/master_experiments/drugflow_main \
    --cpu 4
"""

from __future__ import annotations
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem
from rdkit.Chem import AllChem

# Path config
PDB_BASE = Path("/root/autodl-tmp/data/PDB/P-L")
POCKET_YEARS = {"3mfw":"2001-2010","2gni":"2001-2010","6o4x":"2011-2019",
                "2jke":"2001-2010","2gqn":"2001-2010","6phx":"2011-2019"}

def compute_box_from_ligand(ligand_sdf, padding=5.0):
    """Compute docking box from reference ligand bounding box + padding."""
    mol = Chem.SDMolSupplier(str(ligand_sdf))[0]
    if mol is None:
        return (0,0,0), (22.5,22.5,22.5)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    center = tuple(coords.mean(axis=0).tolist())
    extent = coords.max(axis=0) - coords.min(axis=0)
    box = tuple(max(extent[i] + 2*padding, 20.0) for i in range(3))
    return center, box

def prepare_receptor_pdbqt(protein_pdb, output_pdbqt):
    """Prepare receptor PDBQT using meeko. Handles multi-fragment pockets."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        mol = Chem.MolFromPDBFile(str(protein_pdb), removeHs=True,
                                   sanitize=False)
        if mol is None:
            return False
        # For multi-fragment pocket PDB, take largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        mol = Chem.AddHs(mol, addCoords=True)
        preparator = MoleculePreparation()
        mols = list(preparator.prepare(mol))
        if mols:
            result = PDBQTWriterLegacy.write_string(mols[0])
            pdbqt_str = result[0]
            # Strip ROOT/ENDROOT/BRANCH for rigid receptor (Vina requirement)
            rigid_lines = [l for l in pdbqt_str.split('\n')
                          if l.startswith(('ATOM','HETATM','REMARK'))]
            Path(output_pdbqt).write_text('\n'.join(rigid_lines))
            return True
    except Exception as e:
        print(f"    receptor prep failed: {e}")
    return False

def prepare_ligand_pdbqt(sdf_path, output_pdbqt, mol_name="ligand"):
    """Convert SDF to PDBQT with proper H addition and Gasteiger charges."""
    try:
        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
        if mol is None:
            return False, "SDMolSupplier returned None"

        # Add hydrogens and compute 3D coords
        mol = Chem.AddHs(mol, addCoords=True)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)

        # MMFF94 optimize then compute Gasteiger charges
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
        AllChem.ComputeGasteigerCharges(mol)

        # Write to temp PDB, then convert with meeko
        tmp_pdb = str(output_pdbqt).replace(".pdbqt", "_tmp.pdb")
        Chem.MolToPDBFile(mol, tmp_pdb)

        from meeko import MoleculePreparation, PDBQTWriterLegacy
        mol_pdb = Chem.MolFromPDBFile(tmp_pdb, removeHs=False)
        if mol_pdb is None:
            Path(tmp_pdb).unlink(missing_ok=True)
            return False, "PDB readback failed"

        preparator = MoleculePreparation()
        mols = list(preparator.prepare(mol_pdb))
        if mols:
            result = PDBQTWriterLegacy.write_string(mols[0])
            pdbqt_str = result[0]
            Path(output_pdbqt).write_text(pdbqt_str)
            Path(tmp_pdb).unlink(missing_ok=True)
            return True, None
        Path(tmp_pdb).unlink(missing_ok=True)
        return False, "meeko preparation returned empty"
    except Exception as e:
        return False, str(e)[:100]

def dock_ligand(receptor_pdbqt, ligand_pdbqt, output_pdbqt, center, box_size,
                exhaustiveness=8, n_modes=5, cpu=4):
    """Run AutoDock Vina docking. Returns list of affinity scores."""
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--out", str(output_pdbqt),
        "--num_modes", str(n_modes),
        "--exhaustiveness", str(exhaustiveness),
        "--cpu", str(cpu),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, result.stderr[-200:]
        # Parse scores: Vina table lines have format "   1       -6.2    0.0    0.0"
        # Only accept lines where col 1 is a negative float (affinity), not mode number
        scores = []
        for line in result.stdout.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 3:  # at least: mode_num, affinity, rmsd_lb, rmsd_ub
                try:
                    # affinity is in column 2 (index 1) — always negative for binding
                    s = float(parts[1])
                    if -50 < s < 0:  # binding affinity must be negative
                        scores.append(s)
                except ValueError:
                    continue
        return scores, None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)[:100]

def run_pocket(pocket, base_dir, protein_pdb=None, ref_ligand=None,
               cpu=4, exhaustiveness=8, output_json=None):
    """Dock all valid molecules for one pocket."""
    base = Path(base_dir)
    year = POCKET_YEARS[pocket]

    if protein_pdb is None:
        protein_pdb = PDB_BASE / year / pocket / f"{pocket}_pocket.pdb"  # pocket only (faster)
    if ref_ligand is None:
        ref_ligand = PDB_BASE / year / pocket / f"{pocket}_ligand.sdf"

    print(f"Pocket: {pocket}")
    print(f"  Protein: {protein_pdb}")
    print(f"  Ref ligand: {ref_ligand}")

    # Box setup
    center, box_size = compute_box_from_ligand(ref_ligand)
    print(f"  Box center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"  Box size:   ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f})")

    # Prepare receptor once
    tmpdir = Path(tempfile.mkdtemp(prefix="vina_"))
    receptor_pdbqt = tmpdir / "receptor.pdbqt"
    if not prepare_receptor_pdbqt(protein_pdb, receptor_pdbqt):
        print("  FATAL: receptor preparation failed")
        return {}

    results = {}
    t_start = time.time()
    n_total = 0
    n_success = 0

    for condition in ["baseline", "hard_fix", "com_projection"]:
        sdf_dir = base / pocket / condition / "sdfs"
        if not sdf_dir.exists():
            print(f"  {condition}: no SDFs")
            continue

        sdfs = sorted(sdf_dir.glob("*.sdf"))
        sdfs = [s for s in sdfs if s.stat().st_size > 0]
        if not sdfs:
            print(f"  {condition}: 0 SDFs")
            continue

        # Filter valid-only molecules
        valid_sdfs = []
        for sdf in sdfs:
            try:
                mol = Chem.SDMolSupplier(str(sdf), sanitize=True)[0]
                if mol is not None:
                    valid_sdfs.append(sdf)
            except:
                pass

        if not valid_sdfs:
            print(f"  {condition}: 0 valid SDFs")
            results[condition] = {"vina_mean": None, "vina_std": None,
                                   "n_docked": 0, "n_total": len(sdfs)}
            continue

        scores = []
        errors = 0
        cond_dir = tmpdir / condition
        cond_dir.mkdir(exist_ok=True)

        for i, sdf in enumerate(valid_sdfs):
            lig_pdbqt = cond_dir / f"lig_{i}.pdbqt"
            out_pdbqt = cond_dir / f"out_{i}.pdbqt"

            ok, err = prepare_ligand_pdbqt(sdf, lig_pdbqt)
            if not ok:
                errors += 1
                continue

            mol_scores, dock_err = dock_ligand(
                receptor_pdbqt, lig_pdbqt, out_pdbqt,
                center, box_size, exhaustiveness, cpu=cpu)

            if mol_scores and len(mol_scores) > 0:
                scores.append(min(mol_scores))  # best (most negative) score
                n_success += 1
            else:
                errors += 1

        n_total += len(valid_sdfs)
        if scores:
            results[condition] = {
                "vina_mean": float(np.mean(scores)),
                "vina_std": float(np.std(scores)),
                "vina_best": float(np.min(scores)),
                "vina_worst": float(np.max(scores)),
                "n_docked": len(scores),
                "n_errors": errors,
                "n_total_valid": len(valid_sdfs),
            }
            print(f"  {condition}: Vina={np.mean(scores):.1f}±{np.std(scores):.1f} "
                  f"(n={len(scores)}, errors={errors})")
        else:
            results[condition] = {"vina_mean": None, "error": "all failed",
                                   "n_total_valid": len(valid_sdfs)}
            print(f"  {condition}: ALL DOCKING FAILED ({errors} errors)")

    elapsed = time.time() - t_start
    print(f"  Done: {n_success}/{n_total} docked in {elapsed:.0f}s")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Save
    if output_json:
        out = {"pocket": pocket, "center": center, "box_size": box_size,
               "elapsed_s": elapsed, "conditions": results}
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  Saved: {output_json}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", default="3mfw")
    parser.add_argument("--base-dir", default=str(ROOT / "experiments/master_experiments/drugflow_main"))
    parser.add_argument("--protein-pdb", default=None)
    parser.add_argument("--ref-ligand", default=None)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    run_pocket(args.pocket, args.base_dir, args.protein_pdb, args.ref_ligand,
               args.cpu, args.exhaustiveness, args.output_json)
