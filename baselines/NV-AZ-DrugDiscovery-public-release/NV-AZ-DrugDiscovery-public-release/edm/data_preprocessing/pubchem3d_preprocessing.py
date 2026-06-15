from glob import glob

import gzip
import sys
import pickle

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from natsort import natsorted
import numpy as np
import os

RDLogger.DisableLog("rdApp.*")


def lipinski_rule_of_five(mol):
    vals = []
    vals.append(Descriptors.MolWt(mol) <= 500)
    vals.append(Crippen.MolLogP(mol) <= 5)
    vals.append(Descriptors.NumHDonors(mol) <= 5)
    vals.append(Descriptors.NumHAcceptors(mol) <= 10)
    return all(vals)


def is_3d(mol):
    if mol.GetNumConformers() == 0:
        return False

    conf = mol.GetConformer()
    if not conf.Is3D():
        return False

    if (conf.GetPositions()[:, 2] == 0).all():
        return False
    return True


def all_hs(mol):
    mol_with_hs = Chem.AddHs(mol)
    return mol.GetNumAtoms() == mol_with_hs.GetNumAtoms()


class DataProcessing:
    def __init__(self, save_path, suffix=""):
        self.save_path = save_path
        self.suffix = suffix
        self.atom_types = set()
        self.charges = set()
        self.bond_types = set()
        self.cache = {}
        self.max_size = 500 * 1024 * 1024
        self.file_count = {}
        os.makedirs(self.save_path, exist_ok=True)

    def mol2repr(self, mol):
        conf = mol.GetConformer()
        X = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
        C = np.zeros((mol.GetNumAtoms(), 2), dtype=np.int8)
        A = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.uint8)

        atom_types = set()
        charges = set()
        bond_types = set()

        for atom in mol.GetAtoms():
            aid = atom.GetIdx()
            atype = atom.GetAtomicNum()
            pos = conf.GetAtomPosition(aid)
            charge = atom.GetFormalCharge()
            X[aid] = [pos.x, pos.y, pos.z]
            C[aid] = [atype, charge]
            atom_types.add(atype)
            charges.add(charge)

        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                btype = 1
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                btype = 2
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                btype = 3
            elif bond.GetBondType() == Chem.BondType.AROMATIC:
                btype = 4
            else:
                # SKIP mol
                return
            A[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = btype
            bond_types.add(btype)
        self.atom_types |= atom_types
        self.charges |= charges
        self.bond_types |= bond_types
        self.update(X, C, A + A.T)

    def update(self, X, C, A):
        if len(X) not in self.cache:
            self.cache[len(X)] = []
            os.makedirs(os.path.join(self.save_path, f"{len(X):03d}"), exist_ok=True)

        if len(X) not in self.file_count:
            self.file_count[len(X)] = 0

        self.cache[len(X)].append((X, C, A))

        if (len(self.cache[len(X)]) * len(X) * (4 * 3 + 1 * 2)) >= self.max_size:
            self.save(len(X))

    def save(self, n_atoms):
        print("Save", n_atoms)
        suffix = self.suffix
        X = np.array([x[0] for x in self.cache[n_atoms]])
        C = np.array([x[1] for x in self.cache[n_atoms]])
        A = np.array([x[2] for x in self.cache[n_atoms]])

        fpX = np.memmap(
            os.path.join(
                self.save_path,
                f"{n_atoms:03d}/{self.file_count[n_atoms]:09d}_X_{len(X):d}{suffix}.ndat",
            ),
            dtype=np.float32,
            mode="w+",
            shape=(len(X), n_atoms, 3),
        )
        fpX[:] = X

        fpC = np.memmap(
            os.path.join(
                self.save_path,
                f"{n_atoms:03d}/{self.file_count[n_atoms]:09d}_C_{len(X):d}{suffix}.ndat",
            ),
            dtype=np.int8,
            mode="w+",
            shape=(len(X), n_atoms, 2),
        )
        fpC[:] = C

        fpA = np.memmap(
            os.path.join(
                self.save_path,
                f"{n_atoms:03d}/{self.file_count[n_atoms]:09d}_A_{len(X):d}{suffix}.ndat",
            ),
            dtype=np.uint8,
            mode="w+",
            shape=(len(X), n_atoms, n_atoms),
        )
        fpA[:] = A

        fpX.flush()
        fpA.flush()
        fpC.flush()

        del fpA, fpX, fpC

        self.cache[n_atoms] = []
        self.file_count[n_atoms] += 1

    def save_attributes(self):
        suffix = self.suffix
        with open(
            os.path.join(self.save_path, f"attribute_{suffix}.pkl"), "wb"
        ) as fobj:
            pickle.dump(
                {
                    "atom_types": self.atom_types,
                    "charges": self.charges,
                    "bond_types": self.bond_types,
                },
                fobj,
            )


if __name__ == "__main__":
    dataset_path = sys.argv[1]  # Path containining sdf.gz files
    proc_id, n_procs = 0, 1
    if len(sys.argv) == 4:
        proc_id, n_procs = int(sys.argv[1]), int(sys.argv[2])

    all_paths = natsorted(glob(os.path.join(dataset_path, "*.sdf.gz")))
    all_paths = [(p, os.stat(p).st_size) for p in all_paths]
    all_paths = sorted(all_paths, key=lambda x: x[1], reverse=True)
    all_paths = [p[0] for p in all_paths]
    padded_n_paths = int(np.ceil(len(all_paths) / n_procs)) * n_procs

    idx = np.arange(padded_n_paths)
    idx = idx.reshape(-1, n_procs).T.ravel()
    split_idx = np.array_split(idx, n_procs)[proc_id]
    paths = [all_paths[i] for i in split_idx if i < len(all_paths)]
    print("Number of files to process:", len(paths))
    total_size = 0
    dp = DataProcessing("data/pubchem_new", suffix=f"_{proc_id:03d}")
    for p in paths:
        print("Processing", p)
        try:
            with Chem.ForwardSDMolSupplier(gzip.open(p), removeHs=False) as suppl:
                total_size += os.stat(p).st_size
                for mol in suppl:
                    if mol is None:
                        continue
                    if lipinski_rule_of_five(mol) and is_3d(mol) and all_hs(mol):
                        dp.mol2repr(mol)
        except BaseException:
            pass

    for k in dp.cache:
        if len(dp.cache[k]):
            dp.save(k)
    print("Total size", total_size)
    dp.save_attributes()
