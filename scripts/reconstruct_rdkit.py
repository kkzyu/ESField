#!/usr/bin/env python3
"""RDKit-based molecular reconstruction for TargetDiff output.

Bypasses OpenBabel's reconstruction (which is sensitive to atom valence errors)
by using RDKit's EmbedMolecule + alignment to predicted positions.

Usage:
  python scripts/reconstruct_rdkit.py --results-dir experiments/targetdiff_replication/3mfw/unguided
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

# TargetDiff atom type mappings (add_aromatic mode)
ATOM_TYPE_MAP = {
    0: 1,     # H
    1: 6,     # C (non-aromatic)
    2: 6,     # C (aromatic)
    3: 7,     # N (non-aromatic)
    4: 7,     # N (aromatic)
    5: 8,     # O
    6: 8,     # O (aromatic) — rare
    7: 9,     # F
    8: 15,    # P (non-aromatic)
    9: 15,    # P (aromatic)
    10: 16,   # S (non-aromatic)
    11: 16,   # S (aromatic)
    12: 17,   # Cl
}

AROMATIC_MAP = {
    0: False, 1: False, 2: True, 3: False, 4: True,
    5: False, 6: True, 7: False, 8: False, 9: True,
    10: False, 11: True, 12: False,
}


def reconstruct_rdkit(pos, atom_types_idx, add_h=False):
    """Reconstruct molecule using RDKit.

    Args:
        pos: [n_atoms, 3] predicted coordinates
        atom_types_idx: [n_atoms] TargetDiff type indices (0-12)
        add_h: add explicit hydrogens

    Returns:
        RDKit Mol or None
    """
    n_atoms = len(atom_types_idx)
    atomic_nums = [ATOM_TYPE_MAP.get(int(t), 6) for t in atom_types_idx]
    aromatics = [AROMATIC_MAP.get(int(t), False) for t in atom_types_idx]

    # Create editable mol and add atoms
    mol = Chem.RWMol()
    for an in atomic_nums:
        atom = Chem.Atom(int(an))
        mol.AddAtom(atom)

    # Set aromatic flags
    for i, is_arom in enumerate(aromatics):
        if is_arom:
            mol.GetAtomWithIdx(i).SetIsAromatic(True)

    # Try to get a valid conformation
    try:
        mol = mol.GetMol()
        mol = Chem.Mol(mol)

        # Use ETKDG to embed
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            # Try without ETKDG
            status = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if status != 0:
            return None

        # Align to predicted positions
        conf = mol.GetConformer()
        # Create a reference conformer from predicted positions
        ref_pos = np.array(pos)
        # Simple Kabsch alignment
        rd_pos = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])
        # Center both
        rd_center = rd_pos.mean(axis=0)
        ref_center = ref_pos.mean(axis=0)
        rd_pos_c = rd_pos - rd_center
        ref_pos_c = ref_pos - ref_center
        # SVD for rotation
        H_mat = rd_pos_c.T @ ref_pos_c
        U, _, Vt = np.linalg.svd(H_mat)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        # Apply rotation + translation
        aligned = rd_pos_c @ R.T + ref_center
        for i in range(n_atoms):
            conf.SetAtomPosition(i, aligned[i].tolist())

        # Sanitize
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        return mol

    except Exception as e:
        return None


def reconstruct_batch(results_dir, output_sdf=None):
    """Reconstruct all molecules from a results.pt file."""
    results_path = Path(results_dir) / "results.pt"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        return []

    data = torch.load(results_path, map_location='cpu', weights_only=False)
    positions = data['positions']
    types = data['types']

    print(f"Reconstructing {len(positions)} molecules from {results_path}")

    valid = []
    for i, (pos, v) in enumerate(zip(positions, types)):
        pos64 = pos.astype(np.float64) if pos.dtype != np.float64 else pos
        mol = reconstruct_rdkit(pos64, v)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if '.' not in smiles:
                mol.SetProp("_Name", f"mol_{i:03d}")
                valid.append({"mol": mol, "idx": i, "smiles": smiles, "pos": pos64, "v": v})

    print(f"  Valid: {len(valid)}/{len(positions)} ({len(valid)/max(len(positions),1):.1%})")

    # Save SDF files
    if output_sdf:
        sdf_dir = Path(output_sdf)
        sdf_dir.mkdir(parents=True, exist_ok=True)
        for v in valid:
            Chem.MolToMolFile(v["mol"], str(sdf_dir / f"mol_{v['idx']:03d}.sdf"))

    # Save valid list
    torch.save(valid, Path(results_dir) / "valid_mols.pt")
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-sdf", default=None)
    args = parser.parse_args()

    valid = reconstruct_batch(args.results_dir, args.output_sdf)

    if valid:
        print(f"\nSample SMILES:")
        for v in valid[:5]:
            print(f"  mol_{v['idx']:03d}: {v['smiles']}")


if __name__ == "__main__":
    main()
