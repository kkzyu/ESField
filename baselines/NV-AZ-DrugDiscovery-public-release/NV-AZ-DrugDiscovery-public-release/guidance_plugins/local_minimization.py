import sys
import warnings


from openbabel import openbabel as ob
from openbabel import pybel
import numpy as np

warnings.filterwarnings("ignore", module="openbabel")
ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
np.set_printoptions(precision=3, suppress=True)


def get_ligands(ligand_path, remove_hs=False):
    obconversion = ob.OBConversion()
    obconversion.SetInFormat("sdf")

    obmol = ob.OBMol()
    mols = []
    notatend = obconversion.ReadFile(obmol, ligand_path)
    while notatend:
        mol = pybel.Molecule(obmol)
        if remove_hs:
            mol.removeh()
        mols.append(mol)
        obmol = ob.OBMol()
        notatend = obconversion.Read(obmol)
    return mols


def merge_protein_ligands(
    protein,
    ligand,
    add_ligand_hs=True,
    remove_ligand_isolated_atoms=False,
    th=5.0,
    verbose=False,
):
    lprotein = ob.OBMol(protein)
    lligand = ob.OBMol(ligand)

    ligand_idx_to_protein_idx = {}
    ligand_hs_atom_in_protein_ids = []
    ligand_atom_in_protein_ids = []

    # sanity check
    if verbose:
        natoms = 0
        for atom in ob.OBMolAtomIter(lprotein):
            natoms = natoms + 1
        print("Protein atoms:", natoms)

    ligand_atom_ids, ligand_added_hs = set(), set()
    for atom in ob.OBMolAtomIter(lligand):
        ligand_atom_ids.add(atom.GetId())

    if add_ligand_hs:
        lligand.AddHydrogens()
        for atom in ob.OBMolAtomIter(lligand):
            if atom.GetId() not in ligand_atom_ids:
                ligand_added_hs.add(atom.GetId())

    # store ligand coordinates
    X = []
    for atom in ob.OBMolAtomIter(lligand):
        X.append((atom.GetX(), atom.GetY(), atom.GetZ()))
    X = np.array(X)

    # combine ligand atom into protein
    for atom in ob.OBMolAtomIter(lligand):
        a = lprotein.NewAtom()
        a.SetAtomicNum(atom.GetAtomicNum())
        a.SetVector(atom.GetX(), atom.GetY(), atom.GetZ())
        a.SetFormalCharge(atom.GetFormalCharge())
        ligand_idx_to_protein_idx[atom.GetIdx()] = a.GetIdx()
        if atom.GetId() in ligand_atom_ids:
            ligand_atom_in_protein_ids.append(a.GetId())
        else:
            ligand_hs_atom_in_protein_ids.append(a.GetId())

    for bond in ob.OBMolBondIter(lligand):
        ia = ligand_idx_to_protein_idx[bond.GetBeginAtomIdx()]
        oa = ligand_idx_to_protein_idx[bond.GetEndAtomIdx()]
        order = bond.GetBondOrder()
        lprotein.AddBond(ia, oa, order)

    # remove protein atoms too far away from the ligand
    to_remove = []
    for atom in ob.OBMolAtomIter(lprotein):
        x = np.array((atom.GetX(), atom.GetY(), atom.GetZ()))[None]
        D = np.min(np.sum((x - X) ** 2, axis=-1) ** 0.5)
        if D > th:
            to_remove.append(atom)
    for atom in to_remove:
        lprotein.DeleteAtom(atom)

    # clean up pocket by removing isolated atoms
    to_remove = []
    for atom in ob.OBMolAtomIter(lprotein):
        if atom.HighestBondOrder() < 1:
            if atom.GetId() in ligand_hs_atom_in_protein_ids:
                if remove_ligand_isolated_atoms:
                    to_remove.append(atom)
                    ligand_hs_atom_in_protein_ids.remove(atom.GetId())
                    if verbose:
                        print("WARNING! Isolated hs in the ligand")
            elif atom.GetId() in ligand_atom_in_protein_ids:
                if remove_ligand_isolated_atoms:
                    to_remove.append(atom)
                    ligand_atom_in_protein_ids.remove(atom.GetId())
                    if verbose:
                        print("WARNING! Isolated atom in theligand")
            else:
                to_remove.append(atom)
    for atom in to_remove:
        lprotein.DeleteAtom(atom)
    return lprotein, ligand_hs_atom_in_protein_ids, ligand_atom_in_protein_ids


def optimize(pocket_ligand, atom_ids, nsteps=500, verbose=False):
    lpocket_ligand = ob.OBMol(pocket_ligand)
    constraints = ob.OBFFConstraints()
    for atom in ob.OBMolAtomIter(lpocket_ligand):
        atom_id = atom.GetId()
        # optimize only atoms in atom_ids
        if atom_id not in atom_ids:
            constraints.AddAtomConstraint(atom.GetIdx())

    forcefield = ob.OBForceField.FindForceField("MMFF94")
    setup_status = forcefield.Setup(lpocket_ligand, constraints)
    forcefield.ConjugateGradients(nsteps)
    opt_status = forcefield.GetCoordinates(lpocket_ligand)
    if verbose:
        print(setup_status & opt_status)
    return lpocket_ligand


def extract_ligand(pocket_ligand, atom_ids):
    lligand = ob.OBMol(pocket_ligand)
    to_remove = []
    for atom in ob.OBMolAtomIter(lligand):
        if atom.GetId() not in atom_ids:
            to_remove.append(atom)
    for atom in to_remove:
        lligand.DeleteAtom(atom)
    return lligand


if __name__ == "__main__":
    protein_path = sys.argv[1]
    ligand_path = sys.argv[2]

    protein = next(pybel.readfile("pdb", protein_path))
    ligands = get_ligands(ligand_path)
    opt_ligands = []

    for ligand in ligands:
        if ligand is None:
            continue
        pocket_ligand, ligand_hs_ids, ligand_non_hs_ids = merge_protein_ligands(
            protein.OBMol, ligand.OBMol
        )
        opt_pocket_ligand = optimize(
            pocket_ligand, ligand_hs_ids + ligand_non_hs_ids, verbose=False
        )

        opt_ligand = extract_ligand(
            opt_pocket_ligand, ligand_hs_ids + ligand_non_hs_ids
        )
        opt_ligand = pybel.Molecule(opt_ligand)
        opt_ligand.title = ligand.title
        opt_ligands.append(opt_ligand)

    with pybel.Outputfile("sdf", sys.argv[3], overwrite=True) as sdf_writer:
        for opt_ligand in opt_ligands:
            sdf_writer.write(opt_ligand)
