from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Geometry import Point3D
import torch

from src.utils.torch_forcefield import TorchMMFF94

if __name__ == "__main__":
    protein_path = Path(sys.argv[1])
    ligand_path = Path(sys.argv[2])
    output_path = sys.argv[3]
    device = "cuda"
    pid = ligand_path.parent.name

    protein = Chem.MolFromPDBFile(protein_path, removeHs=False, sanitize=False)
    protein = Chem.AddHs(protein)
    Chem.GetSSSR(protein)
    ff = TorchMMFF94(protein=protein, device=device)

    new_mols = []
    try:
        with Chem.SDMolSupplier(ligand_path, sanitize=False, removeHs=False) as suppl:
            for mid, mol in enumerate(suppl):
                _mol = Chem.Mol(mol)
                _mol.UpdatePropertyCache()
                Chem.GetSSSR(_mol)
                conf = _mol.GetConformer()
                x = torch.tensor(
                    conf.GetPositions(),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                try:
                    ff.setup(_mol)
                    opt = torch.optim.Adam([x], lr=0.1)
                    for i in range(500):
                        opt.zero_grad()
                        loss, ligand_ff = ff.forward(x)
                        loss.backward()
                        opt.step()
                        if i == 0:
                            print(pid, mid, loss.item(), ligand_ff)
                        if i == 499:
                            print(pid, mid, loss.item(), ligand_ff)
                    for i, p in enumerate(x.detach().cpu().numpy()):
                        conf.SetAtomPosition(i, Point3D(*p.tolist()))
                    new_mols.append(_mol)
                except Exception:
                    new_mols.append(mol)
    except Exception:
        pass
    writer = Chem.SDWriter(output_path)
    for m in new_mols:
        writer.write(m)
    writer.close()
