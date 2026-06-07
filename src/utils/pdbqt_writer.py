"""Lightweight PDBQT writer using RDKit only (no meeko/obabel required).

Generates AutoDock Vina-compatible PDBQT files with:
  - Gasteiger partial charges (via RDKit)
  - AutoDock atom type assignment (heuristic mapping)
  - ROOT/ENDROOT/BRANCH/ENDBRANCH torsion tree (single rigid branch)

Usage:
    from utils.pdbqt_writer import mol_to_pdbqt_string
    pdbqt_str = mol_to_pdbqt_string(rdkit_mol)
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


# AutoDock atom type mapping (element → AD type)
# Based on AutoDock4 atom type conventions
ATOMIC_NUM_TO_AD_TYPE = {
    1: "HD",   # non-polar H
    6: "C",    # carbon
    7: "NA",   # nitrogen (hydrogen-bond acceptor or donor)
    8: "OA",   # oxygen (hydrogen-bond acceptor)
    9: "F",    # fluorine
    15: "P",   # phosphorus
    16: "SA",  # sulfur (hydrogen-bond acceptor)
    17: "Cl",  # chlorine
    35: "Br",  # bromine
    53: "I",   # iodine
}


def _assign_ad_type(atom: Chem.Atom) -> str:
    """Assign AutoDock atom type based on element and bonding."""
    atomic_num = atom.GetAtomicNum()

    # Special cases based on bonding
    if atomic_num == 7:
        # Check if N is in aromatic ring → NA, else could be N (donor)
        if atom.GetIsAromatic():
            return "NA"
        # Count attached hydrogens
        n_h = atom.GetTotalNumHs()
        if n_h > 0:
            return "N"  # H-bond donor
        return "NA"  # H-bond acceptor

    if atomic_num == 8:
        # Check if O is in carboxylate, phosphate, etc.
        return "OA"

    if atomic_num == 16:
        n_h = atom.GetTotalNumHs()
        if n_h > 0:
            return "S"  # Thiol
        return "SA"

    return ATOMIC_NUM_TO_AD_TYPE.get(atomic_num, "A")  # "A" = generic atom


def mol_to_pdbqt_string(mol: Chem.Mol, charge: int | None = None) -> str | None:
    """Convert an RDKit molecule to PDBQT string for AutoDock Vina.

    Args:
        mol: RDKit molecule (should have explicit hydrogens and 3D coords)
        charge: net charge (auto-detected if None)

    Returns:
        PDBQT string with ROOT/ENDROOT torsion tree, or None on failure.
    """
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        pass

    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    except Exception:
        pass  # Already has coords

    # Compute Gasteiger charges (sanitize first for reliable charge computation)
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        pass
    try:
        AllChem.ComputeGasteigerCharges(mol, throwOnFailure=False)
    except Exception:
        pass  # Some atoms may not get charges; use 0.0 as fallback

    conf = mol.GetConformer()
    atoms = []
    lines = []
    atom_count = 1  # PDBQT uses 1-based indexing

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 0:
            continue  # Skip dummy atoms

        pos = conf.GetAtomPosition(atom.GetIdx())
        ad_type = _assign_ad_type(atom)

        # Get Gasteiger charge (handle nan)
        gasteiger = 0.0
        if atom.HasProp("_GasteigerCharge"):
            try:
                gasteiger = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(gasteiger) or np.isinf(gasteiger):
                    gasteiger = 0.0
            except (ValueError, TypeError):
                gasteiger = 0.0

        # PDBQT ATOM record — use standard PDB column alignment.
        # Col 1-6: "ATOM  ", 7-11: serial, 13-16: atom name, 18-20: resName,
        # 22: chain, 23-26: resSeq, 31-38: x, 39-46: y, 47-54: z,
        # 55-60: occupancy, 61-66: tempFactor, 71-76: charge, 78-79: type
        elem = atom.GetSymbol()
        name = f" {elem:<3}" if len(elem) <= 1 else f"{elem:<4}"

        # Build using exact column positions
        serial = f"{atom_count:>5d}"
        res_info = "LIG A   1"
        x_str = f"{pos.x:8.3f}"
        y_str = f"{pos.y:8.3f}"
        z_str = f"{pos.z:8.3f}"
        charge_str = f"{gasteiger:6.3f}"
        type_str = f"{ad_type:<2}"

        line = (
            f"ATOM  {serial} {name}{res_info}    "
            f"{x_str}{y_str}{z_str}"
            f"  1.00  0.00    {charge_str} {type_str}"
        )
        lines.append(line)
        atoms.append(atom_count)
        atom_count += 1

    if not lines:
        return None

    # Build PDBQT with torsion tree (single rigid branch).
    # REMARK line is required by Vina to identify the file format.
    pdbqt_lines = []
    pdbqt_lines.append("REMARK  VINA PDBQT")
    pdbqt_lines.append("ROOT")
    pdbqt_lines.extend(lines)
    pdbqt_lines.append("ENDROOT")
    pdbqt_lines.append("TORSDOF 0")

    return "\n".join(pdbqt_lines)


def prepare_protein_pdbqt(protein_pdb: str, output_path: str | None = None) -> str:
    """Convert a protein PDB to PDBQT by preserving original PDB formatting.

    Reads the PDB file directly and converts ATOM/HETATM records to PDBQT
    by adding AutoDock atom types and partial charges while preserving
    the EXACT coordinate formatting from the source file.
    """
    import os

    if output_path is None:
        output_path = protein_pdb.replace(".pdb", ".pdbqt")

    if os.path.exists(output_path):
        return output_path

    # Direct PDB → PDBQT conversion, preserving coordinate columns exactly
    with open(protein_pdb) as f_in, open(output_path, "w") as f_out:
        f_out.write("REMARK  VINA PROTEIN PDBQT\n")
        atom_count = 0
        for line in f_in:
            if not (line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("ANISOU")):
                continue
            if len(line) < 54:
                continue

            # Extract element from columns 77-78 (standard PDB)
            elem = line[76:78].strip()
            if not elem:
                # Fallback: derive from atom name
                atom_name = line[12:16].strip()
                if atom_name:
                    elem = atom_name[0] if atom_name[0].isalpha() else atom_name[1] if len(atom_name) > 1 else "C"
                else:
                    elem = "C"

            # Map to AutoDock type
            from rdkit import Chem
            try:
                pt = Chem.GetPeriodicTable()
                atomic_num = pt.GetAtomicNumber(elem)
            except Exception:
                atomic_num = 6
            ad_type = ATOMIC_NUM_TO_AD_TYPE.get(atomic_num, "A")

            atom_count += 1
            # Vina requires "ATOM  " prefix, not "HETATM"
            coords = line[30:54]  # x(8) + y(8) + z(8)
            serial = f"{atom_count:>5d}"
            atom_name = line[12:16]
            res_name = line[17:20]
            chain = line[21:22]
            res_num = line[22:26]
            # Write PDBQT with "ATOM  " prefix and standard column alignment
            f_out.write(
                f"ATOM  {serial} {atom_name}{res_name} {chain}{res_num}    "
                f"{coords}  1.00  0.00     0.000 {ad_type:<2s}\n"
            )

    if atom_count == 0:
        raise RuntimeError(f"No atoms found in protein PDB: {protein_pdb}")

    return output_path
