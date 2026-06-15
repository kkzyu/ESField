from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import numpy as np
import torch


def cloud2mol(x, c, a):
    mol = AllChem.RWMol()
    for atom_num, charge in c:
        atom = Chem.Atom(int(atom_num))
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)

    frontier = set()
    for i, j in np.vstack(np.where(a)).T:
        if i == j:
            continue
        if (i, j) in frontier:
            continue
        if (j, i) in frontier:
            continue
        if a[i, j] == 1:
            bt = Chem.BondType.SINGLE
        elif a[i, j] == 2:
            bt = Chem.BondType.DOUBLE
        elif a[i, j] == 3:
            bt = Chem.BondType.TRIPLE
        elif a[i, j] == 4:
            bt = Chem.BondType.AROMATIC
        else:
            continue

        mol.AddBond(int(i), int(j), order=bt)
        frontier.add((i, j))
        frontier.add((j, i))

    conf = Chem.Conformer()
    for i, p in enumerate(x):
        point = Point3D(float(p[0]), float(p[1]), float(p[2]))
        conf.SetAtomPosition(i, point)
    mol.AddConformer(conf)
    mol = Chem.Mol(mol)
    return mol


def to_rdkit(x, c, a):
    X = np.vstack(x)
    C = np.vstack(c)
    A = np.vstack(a)
    mols = []
    eg = []
    for x, c, a in zip(X, C, A):
        mol = cloud2mol(x, c, a)
        # mol = pos2mol(x, c[...,0], c[...,1])
        try:
            mol.UpdatePropertyCache()
            Chem.GetSSSR(mol)
            if len(Chem.GetMolFrags(mol)) > 1:
                mol = None
            if mol is not None:
                _mol = Chem.Mol(mol)
                Chem.SanitizeMol(_mol)
        except Exception:
            mol = None

        if mol is not None:
            mols.append(mol)
        else:
            continue

        try:
            mmprop = AllChem.MMFFGetMoleculeProperties(mol)
            mmff94 = AllChem.MMFFGetMoleculeForceField(mol, mmprop)
            e = mmff94.CalcEnergy()
            k = mol.GetNumAtoms()
            eg.append(e / float(k))
            # mmff94.Minimize()
            # egm.append(mmff94.CalcEnergy()  / float(k))
        except Exception:
            pass
    return mols, eg


def prepare_input(batch, t, vocab):
    h = torch.nn.functional.one_hot(
        batch["c"][..., 0], num_classes=len(vocab.atom_types) + 1
    )
    h2 = torch.nn.functional.one_hot(
        batch["c"][..., 1], num_classes=len(vocab.charge_types) + 1
    )
    # h = torch.cat((h1, h2), -1)
    e = torch.nn.functional.one_hot(batch["a"], num_classes=len(vocab.bond_types))
    t = t.view(-1, 1, 1).expand(-1, h.size(1), -1)
    features = torch.cat((t, h.float()), dim=2)
    return batch["x"], features, h2, e, batch["mask"]


def filter_based_on_atomtype(batch_data, vocab):

    filtered_data = {}
    known_types = torch.tensor(list(vocab.atom_num_types))
    type_mask = torch.isin(batch_data["c"][:, :, 0], known_types)
    for k in batch_data:
        if k == "a":
            if batch_data["a"]:
                filtered_data[k] = batch_data[k][:, type_mask.squeeze(0)][
                    :, :, type_mask.squeeze(0)
                ]
            else:
                filtered_data["a"] = None
        else:
            filtered_data[k] = batch_data[k][type_mask].unsqueeze(0)

    return filtered_data


def prepare_protein_input(batch, vocab):
    features = torch.nn.functional.one_hot(
        batch["c"][..., 0], num_classes=len(vocab.atom_types) + 1
    )
    h2 = torch.nn.functional.one_hot(
        batch["c"][..., 1], num_classes=len(vocab.charge_types) + 1
    )
    # h = torch.cat((h1, h2), -1)
    # e = torch.nn.functional.one_hot(batch["a"], num_classes=len(vocab.bond_types))
    return batch["x"], features, h2, None, batch["mask"]


def _uniform_sample_step(
    curr_dist, pred_dist, t, step_size, cat_noise=1, eps=1e-5, return_distr=False
):
    n_categories = pred_dist.size(-1)

    curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
    pred_probs_curr = torch.gather(pred_dist, -1, curr)

    # Setup batched time tensor and noise tensor
    ones = [1] * (len(pred_dist.shape) - 1)
    times = t.view(-1, *ones).clamp(min=eps, max=1.0 - eps)
    noise = torch.zeros_like(times)
    noise[times + step_size < 1.0] = cat_noise

    # Off-diagonal step probs
    mult = (1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times)
    first_term = step_size * mult * pred_dist
    second_term = step_size * noise * pred_probs_curr
    step_probs = (first_term + second_term).clamp(max=1.0)

    # On-diagonal step probs
    step_probs.scatter_(-1, curr, 0.0)
    diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    step_probs.scatter_(-1, curr, diags)

    # Sample and convert back to one-hot so that all strategies represent data the same way
    samples = torch.distributions.Categorical(step_probs).sample()
    if return_distr:
        return (
            torch.nn.functional.one_hot(samples, num_classes=n_categories),
            step_probs,
        )
    return torch.nn.functional.one_hot(samples, num_classes=n_categories)
