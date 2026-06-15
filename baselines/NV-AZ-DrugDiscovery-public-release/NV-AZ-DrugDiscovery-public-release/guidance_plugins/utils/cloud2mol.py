import numpy as np
import sys
from typing import Tuple
from typing import Union

from func_timeout import func_timeout
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from openbabel import openbabel as ob
from openbabel import pybel

RDLogger.DisableLog("rdApp.*")
pybel.ob.obErrorLog.StopLogging()


def cloud2mol(
    coords: np.ndarray,
    types: np.ndarray,
    bond_timeout: float = 5.0,
    validate_energy: bool = True,
    sanitize: bool = True,
) -> Tuple[Union[Chem.Mol, None], pybel.Molecule, Union[float, None]]:
    """Convert 3D coordinates and atom types to an RDKit Mol object

    This will attempt to infer the bonds from the coordinates and types.
    If that is not possible None is returned.

    Args:
        coords (np.ndarray): Coordinate tensor, shape [num_atoms, 3]
        types (np.ndarray): Atom types tensor, shape [num_atoms]
        bond_timeout (float): Timeout for Open Babel default
                              bond inference algorithm.
        validate_energy (bool): Add extra check. The compound is valid if
                                RDKit is able to initialize the FF
        sanitize (bool): toggle rdkit sanitization

    Returns:
        rdkit.Chem.Mol: RDKit molecule object with one conformer,
                        or a None object if a mol could not be generated
    """
    _pt = Chem.GetPeriodicTable()
    coord_strs = ["\t".join([f"{c:.6f}" for c in cs]) for cs in coords.tolist()]
    atom_symbols = [_pt.GetElementSymbol(int(atomic)) for atomic in types.tolist()]
    xyz_str_header = f"{str(coords.shape[0])}\n\n"
    xyz_strs = [
        f"{str(atom)}\t{coord_str}" for coord_str, atom in zip(coord_strs, atom_symbols)
    ]
    xyz_str = xyz_str_header + "\n".join(xyz_strs)

    energy = np.inf
    try:
        mol = pybel.readstring("xyz", xyz_str)
        mol_b = func_timeout(bond_timeout, _infer_bonds_, args=(mol,))
        mol_str = mol_b.write("mol")
        rdkit_mol = Chem.MolFromMolBlock(mol_str, removeHs=False, sanitize=sanitize)

        sanity_check_mol = Chem.MolFromSmiles(Chem.MolToSmiles(rdkit_mol))
        if sanity_check_mol is None:
            rdkit_mol = None

        rdkit_mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(rdkit_mol)

        if validate_energy:
            ff_props = AllChem.MMFFGetMoleculeProperties(rdkit_mol)
            ff = AllChem.MMFFGetMoleculeForceField(rdkit_mol, ff_props)
            energy = ff.CalcEnergy()
    except BaseException:
        rdkit_mol = None
    return rdkit_mol, mol_b, energy


def _infer_bonds_(mol: pybel.Molecule) -> pybel.Molecule:
    obmol = mol.OBMol

    obmol.ConnectTheDots()
    ob.OBAromaticTyper().AssignAromaticFlags(obmol)
    obmol.PerceiveBondOrders()

    for atom in ob.OBMolAtomIter(obmol):
        if atom.GetAtomicNum() == 7:  # fix N charge
            atom.SetFormalCharge(atom.GetExplicitValence() - 3)
        elif atom.GetAtomicNum() == 8:  # fix O charge
            atom.SetFormalCharge(atom.GetExplicitValence() - 2)

    pybmol = pybel.Molecule(obmol)
    return pybmol


if __name__ == "__main__":
    point_clouds = np.load(sys.argv[1])

    writer = Chem.SDWriter(sys.argv[2])

    valid = 0
    energys = []
    total = len(point_clouds)

    for i, x in enumerate(point_clouds):
        rdmol, mol_ob, energy = cloud2mol(x[..., :3], x[..., 3], sanitize=False)
        if rdmol is None:
            continue
        rdmol.SetProp("_Name", str(i))
        if energy < np.inf:
            rdmol.SetProp("_Energy", str(energy))
            energys.append(energy)
        else:
            rdmol.SetProp("_Energy", "Infty")
        writer.write(rdmol)
        valid += 1
    writer.close()
    print(f"Saved {valid} valid molecules over {total} molecules in {sys.argv[2]}")
    print(f"mean energy : {np.mean(energys)}")
