from typing import Tuple
from typing import List

import numpy as np
from openbabel import openbabel as ob
from openbabel import pybel

# from guidance_plugins.utils.cloud2mol import cloud2mol
from .utils.cloud2mol import cloud2mol


def compute_energy(
    pybelmol: pybel.Molecule,
    requires_grad: bool = True,
    forcefield_name: str = "mmff94",
) -> Tuple[float, np.ndarray]:
    # Convert Pybel molecule to OBMol
    ob_mol = pybelmol.OBMol

    # Setup the force field
    forcefield = ob.OBForceField.FindForceField(forcefield_name)

    if not forcefield:
        raise ValueError(f"Force field {forcefield_name} not found")
    # Assign force field parameters to the molecule
    forcefield.Setup(ob_mol)
    # Calculate the energy
    energy = forcefield.Energy()
    # energy_thre = 10**4
    # if abs(energy) >= energy_thre:
    #     energy = np.nan
    if np.isinf(energy):
        energy = np.nan
    n_atom = ob_mol.NumAtoms()
    gradient = np.zeros((n_atom, 3))
    if (not np.isnan(energy)) and requires_grad:
        for i in range(n_atom):
            atom = ob_mol.GetAtom(i + 1)  # OBMol atoms are 1-indexed
            force = forcefield.GetGradient(atom)
            force_xyz = np.array([force.GetX(), force.GetY(), force.GetZ()])
            except_bool = np.logical_or(np.isnan(force_xyz), np.isinf(force_xyz))
            # skip applying energy guidance for the entire molecule
            # if grad contains nan or inf
            if except_bool.any():
                gradient = np.zeros((n_atom, 3))
                break
            else:
                gradient[i, :] = force_xyz
    return energy, gradient


def score(
    xt_pos: np.ndarray, xt_cat: np.ndarray, requires_grad: bool = True
) -> Tuple[List[float], np.ndarray]:
    """
    xt_pos : batch x num_atom x 3
    xt_cat : batch x num_atom x 1
    grad_batch : batch x num_atom x 3
    """
    energy_batch = []
    grad_batch = np.zeros(xt_pos.shape)
    for i, (pos, cat) in enumerate(zip(xt_pos, xt_cat)):
        rdmol, ob_mol, __ = cloud2mol(pos, cat.squeeze(-1), sanitize=False)
        if rdmol:
            energy, grad = compute_energy(ob_mol, requires_grad=requires_grad)
            energy_batch.append(energy)
            if requires_grad:
                grad_batch[i, :, :] = grad
        else:
            energy_batch.append(np.nan)
    return energy_batch, grad_batch
