from typing import Tuple
from typing import List

from openbabel import openbabel as ob
from openbabel import pybel
import numpy as np

# from guidance_plugins.utils.cloud2mol import cloud2mol
from .utils.cloud2mol import cloud2mol


def merge_protein_ligands(protein, ligand, th=5.0, verbose=False):
    # `th` is the pocket-cutoff distance (Angstroms): protein atoms within
    # this distance of any ligand atom are kept as the binding pocket; the
    # rest of the protein is discarded to keep the force-field evaluation
    # tractable. 5.0 A is the standard pocket-definition radius used across
    # structure-based drug design benchmarks.
    lprotein = ob.OBMol(protein)
    lligand = ob.OBMol(ligand)
    ligand_idx_to_protein_idx = {}
    ligand_protein_atom_ids = []

    # sanity check
    if verbose:
        natoms = 0
        for atom in ob.OBMolAtomIter(lprotein):
            natoms = natoms + 1
        print("Protein atoms:", natoms)

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
        ligand_protein_atom_ids.append(a.GetId())

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
            if atom.GetId() not in ligand_protein_atom_ids:
                # print("WARNING! Removing atom from ligand")
                to_remove.append(atom)
    for atom in to_remove:
        lprotein.DeleteAtom(atom)
    return lprotein, ligand_protein_atom_ids


def compute_energy(
    ligand: pybel.Molecule,
    protein: pybel.Molecule,
    requires_grad: bool = True,
    forcefield_name: str = "mmff94",
    atom_only=False,
    th: float = 5.0,
    steps=1,
) -> Tuple[float, np.ndarray]:
    pocket_ligand, ligand_atom_ids = merge_protein_ligands(
        protein.OBMol, ligand.OBMol, th=th
    )

    # Setup the force field
    forcefield = ob.OBForceField.FindForceField(forcefield_name)

    # energy_terms = ob.OBFF_ENERGY
    # if atom_only:
    #    energy_terms = ob.OBFF_EVDW + ob.OBFF_EELECTROSTATIC
    # Those unfortunately do not work.. OBFF_ENERGY#, OBFF_EBOND, OBFF_EANGLE, OBFF_ESTRBND, OBFF_ETORSION, OBFF_EOOP, OBFF_EVDW, OBFF_EELECTROSTATIC

    if not forcefield:
        raise ValueError(f"Force field {forcefield_name} not found")

    lpocket_ligand = ob.OBMol(pocket_ligand)
    constraints = ob.OBFFConstraints()
    total_atoms = 0
    for atom in ob.OBMolAtomIter(lpocket_ligand):
        total_atoms += 1
        atom_id = atom.GetId()
        # optimize only atoms in atom_ids
        if atom_id not in ligand_atom_ids:
            constraints.AddAtomConstraint(atom.GetIdx())

    forcefield.Setup(lpocket_ligand, constraints)
    if atom_only:
        e_fns = [forcefield.E_Electrostatic, forcefield.E_VDW]
    else:
        e_fns = [forcefield.Energy]

    n_atom = len(ligand_atom_ids)
    gradient = np.zeros((n_atom, 3))
    total_energy = 0.0
    for e_fn in e_fns:
        energy = e_fn()
        if np.isinf(energy):
            energy = np.nan
        total_energy += energy
        if (not np.isnan(energy)) and requires_grad:
            for i, atom_id in enumerate(ligand_atom_ids):
                atom = lpocket_ligand.GetAtomById(atom_id)
                force = forcefield.GetGradient(atom)
                # !!!IMPORTANT!!! openbabel return the negative gradient!!!
                force_xyz = -np.array([force.GetX(), force.GetY(), force.GetZ()])
                except_bool = np.logical_or(np.isnan(force_xyz), np.isinf(force_xyz))
                if except_bool.any():
                    gradient = np.zeros((n_atom, 3))
                    break
                else:
                    gradient[i, :] += force_xyz
    return total_energy, gradient / len(e_fns)


def score(
    xt_pos: np.ndarray,
    xt_cat: np.ndarray,
    protein_path: str,
    requires_grad: bool = True,
    th: float = 5.0,
    atom_only: bool = False,
) -> Tuple[List[float], np.ndarray]:
    """
    xt_pos : batch x num_atom x 3
    xt_cat : batch x num_atom x 1
    grad_batch : batch x num_atom x 3
    """
    protein = next(pybel.readfile("pdb", protein_path))
    energy_batch = []
    grad_batch = np.zeros(xt_pos.shape)
    for i, (pos, cat) in enumerate(zip(xt_pos, xt_cat)):
        rdmol, ob_mol, _ = cloud2mol(pos, cat.squeeze(-1), sanitize=False)
        if rdmol:
            energy, grad = compute_energy(
                ob_mol, protein, th=th, requires_grad=requires_grad, atom_only=atom_only
            )
            energy_batch.append(energy)
            if requires_grad:
                grad_batch[i, :, :] = grad
        else:
            energy_batch.append(np.nan)
    return energy_batch, grad_batch
