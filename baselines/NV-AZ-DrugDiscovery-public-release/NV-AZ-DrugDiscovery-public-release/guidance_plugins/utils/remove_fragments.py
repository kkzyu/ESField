from rdkit import Chem


def remove_fragments(insdf, outsdf):
    writer = Chem.SDWriter(outsdf)
    with Chem.SDMolSupplier(insdf, sanitize=False, removeHs=False) as suppl:
        for mol in suppl:
            name = mol.GetProp("_Name")
            if mol is None:
                continue
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(frags) == 0:
                continue
            frags = [(f.GetNumAtoms(), f) for f in frags]
            frags = sorted(frags, key=lambda x: x[0], reverse=True)
            largest_frag = frags[0][1]
            if largest_frag is None:
                continue
            largest_frag.SetProp("_Name", name)
            writer.write(largest_frag)
    writer.close()


if __name__ == "__main__":
    import sys

    insdf = sys.argv[1]
    outsdf = sys.argv[2]
    remove_fragments(insdf, outsdf)
