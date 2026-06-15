import sys

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.QED import qed

if __name__ == "__main__":
    mols = []

    mol = Chem.SDMolSupplier(sys.argv[1])[0]
    native_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    with Chem.SDMolSupplier(sys.argv[2]) as suppl:
        for mol in suppl:
            if mol is not None:
                mols.append(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

    draw_options = Draw.MolDrawOptions()
    draw_options.legendFontSize = 34
    legends = [f"Native;QED={qed(native_mol):.2f};Frags=1"]
    legends += [
        f"{i:d};QED={qed(mol):.2f};Frags={len(Chem.GetMolFrags(mol)):d}"
        for i, mol in enumerate(mols)
    ]
    img = Draw.MolsToGridImage(
        [native_mol] + mols,
        molsPerRow=5,
        subImgSize=(300, 300),
        legends=legends,
        drawOptions=draw_options,
        returnPNG=False,
    )
    img.save(sys.argv[3])
