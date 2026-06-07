"""MMFF94/UFF molecular geometry minimization (RDKit-only, no external deps).

Provides:
  - minimize_molecule_with_mmff: single-molecule minimization
  - batch_minimize_sdf:         parallel batch SDF minimization

For Phase IIb: pre-minimized molecules give Vina scores in the -10 to +10
kcal/mol range, enabling meaningful occupied-vs-non-occupied comparisons.

Usage:
    from utils.minimize_molecule import minimize_molecule_with_mmff, batch_minimize_sdf
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def minimize_molecule_with_mmff(
    mol: Chem.Mol,
    max_iters: int = 200,
    force_tolerance: float = 0.01,
    add_hydrogens: bool = True,
) -> tuple[Chem.Mol | None, dict[str, Any]]:
    """Minimize a single molecule using MMFF94 (UFF fallback).

    Args:
        mol: RDKit Mol object (with 3D conformer or embeddable)
        max_iters: max optimization iterations
        force_tolerance: convergence threshold for max force component
        add_hydrogens: add explicit hydrogens before minimization

    Returns:
        (minimized_mol, info_dict) where info_dict has:
          - converged: bool
          - energy_before: float (kcal/mol, NaN if not computed)
          - energy_after: float (kcal/mol)
          - ff_name: "MMFF94" or "UFF"
          - error: str or None
    """
    info: dict[str, Any] = {
        "converged": False,
        "energy_before": float("nan"),
        "energy_after": float("nan"),
        "ff_name": "none",
        "error": None,
    }

    try:
        # Work on a copy
        mol = Chem.RWMol(mol)

        # Add hydrogens
        if add_hydrogens:
            mol = Chem.AddHs(mol, addCoords=True)

        # Try embedding if no 3D coords
        conf = mol.GetConformer()
        has_coords = all(
            abs(conf.GetAtomPosition(i).x) > 1e-6 or
            abs(conf.GetAtomPosition(i).y) > 1e-6 or
            abs(conf.GetAtomPosition(i).z) > 1e-6
            for i in range(mol.GetNumAtoms())
        )
        if not has_coords:
            AllChem.EmbedMolecule(mol, randomSeed=42)

        # Sanitize (skip kekulization which can fail on weird bonds)
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            pass

        # Try MMFF94 first
        ff = None
        ff_name = ""
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                ff_name = "MMFF94"
        except Exception:
            pass

        # Fallback to UFF
        if ff is None:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol)
                ff_name = "UFF"
            except Exception:
                ff = None

        if ff is None:
            info["error"] = "No force field available"
            return mol, info

        info["ff_name"] = ff_name

        # Compute initial energy
        try:
            info["energy_before"] = float(ff.CalcEnergy())
        except Exception:
            pass

        # Minimize
        result = ff.Minimize(maxIts=max_iters, forceTol=force_tolerance)
        info["converged"] = (result == 0)

        # Final energy
        try:
            info["energy_after"] = float(ff.CalcEnergy())
        except Exception:
            pass

        # Update coordinates in mol
        ff.AddCoordsToMol(mol)

        return mol.GetMol(), info

    except Exception as e:
        info["error"] = str(e)[:200]
        # Try to return sanitized mol even on failure
        try:
            return mol.GetMol() if isinstance(mol, Chem.RWMol) else mol, info
        except Exception:
            return None, info


def _minimize_one(args):
    """Worker function for multiprocessing. (mol, idx, max_iters, force_tol) -> (idx, mol, info)."""
    mol, idx, max_iters, force_tol = args
    try:
        new_mol, info = minimize_molecule_with_mmff(mol, max_iters=max_iters,
                                                     force_tolerance=force_tol)
        return idx, new_mol, info
    except Exception as e:
        return idx, None, {"converged": False, "error": str(e)[:200],
                            "energy_before": float("nan"),
                            "energy_after": float("nan"),
                            "ff_name": "none"}


def batch_minimize_sdf(
    input_sdf: str,
    output_sdf: str,
    max_iters: int = 200,
    force_tolerance: float = 0.01,
    n_jobs: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:
    """Minimize all molecules in an SDF file in parallel.

    Args:
        input_sdf: path to input SDF
        output_sdf: path to output minimized SDF
        max_iters: max MMFF iterations
        force_tolerance: convergence tolerance
        n_jobs: number of parallel workers
        verbose: print progress

    Returns:
        Summary dict with:
          - n_input: total input molecules
          - n_success: successfully minimized
          - n_failed: minimization failed
          - mean_energy_before: float
          - mean_energy_after: float
          - ff_counts: {"MMFF94": N, "UFF": N}
    """
    # Load molecules
    mols = list(Chem.SDMolSupplier(input_sdf, sanitize=False))
    n_input = len(mols)
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_indices]

    if verbose:
        print(f"  Loading {n_input} molecules ({len(valid_mols)} valid)")

    if not valid_mols:
        return {"n_input": n_input, "n_success": 0, "n_failed": n_input}

    # Parallel minimization
    tasks = [(m, i, max_iters, force_tolerance) for i, m in enumerate(valid_mols)]
    results = {}  # idx -> (mol, info)

    with Pool(processes=n_jobs) as pool:
        for idx, new_mol, info in pool.imap_unordered(_minimize_one, tasks, chunksize=5):
            results[idx] = (new_mol, info)
            if verbose and len(results) % max(1, len(valid_mols) // 4) == 0:
                n_ok = sum(1 for _, info in results.values() if info["converged"])
                print(f"    {len(results)}/{len(valid_mols)} minimized, {n_ok} converged")

    # Reconstruct molecule list in original order
    minimized_mols = [None] * n_input
    all_infos = [None] * n_input
    for result_idx, (new_mol, info) in results.items():
        original_idx = valid_indices[result_idx]
        minimized_mols[original_idx] = new_mol
        all_infos[original_idx] = info

    # Write output SDF with energy properties
    writer = Chem.SDWriter(output_sdf)
    writer.SetKekulize(False)
    n_success = 0
    n_failed = 0
    energy_before_vals = []
    energy_after_vals = []
    ff_counts: dict[str, int] = {}

    for i in range(n_input):
        mol = minimized_mols[i]
        info = all_infos[i]
        if mol is not None:
            try:
                # Set SD properties
                if info:
                    mol.SetProp("MMFF_energy_before", f"{info.get('energy_before', float('nan')):.3f}")
                    mol.SetProp("MMFF_energy_after", f"{info.get('energy_after', float('nan')):.3f}")
                    mol.SetProp("MMFF_ff_name", info.get("ff_name", "none"))
                    mol.SetProp("MMFF_converged", str(info.get("converged", False)))
                    mol.SetProp("MMFF_error", info.get("error") or "")

                    ff_counts[info.get("ff_name", "none")] = \
                        ff_counts.get(info.get("ff_name", "none"), 0) + 1

                    eb = info.get("energy_before", float("nan"))
                    ea = info.get("energy_after", float("nan"))
                    if not np.isnan(eb):
                        energy_before_vals.append(eb)
                    if not np.isnan(ea):
                        energy_after_vals.append(ea)

                writer.write(mol)
                n_success += 1
            except Exception:
                n_failed += 1
        else:
            n_failed += 1

    writer.close()

    summary = {
        "n_input": n_input,
        "n_success": n_success,
        "n_failed": n_failed,
        "mean_energy_before": float(np.mean(energy_before_vals)) if energy_before_vals else float("nan"),
        "mean_energy_after": float(np.mean(energy_after_vals)) if energy_after_vals else float("nan"),
        "ff_counts": ff_counts,
    }

    if verbose:
        print(f"  Done: {n_success} OK, {n_failed} failed")
        print(f"  Energy: {summary['mean_energy_before']:.1f} → "
              f"{summary['mean_energy_after']:.1f} kcal/mol")
        print(f"  Force fields: {ff_counts}")

    return summary
