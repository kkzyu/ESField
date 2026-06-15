from rdkit import Chem
from rdkit import RDLogger
import prolif as plf

RDLogger.DisableLog("rdApp.*")


def get_protein_ligand_interactions(protein_path, ligands_path):
    interactions = []

    rdkit_protein = Chem.MolFromPDBFile(protein_path, removeHs=False, sanitize=False)
    protein_mol = plf.Molecule(rdkit_protein)
    ligand_mols = plf.sdf_supplier(ligands_path)

    for ligand_mol in ligand_mols:
        if ligand_mol is None:
            continue
        try:
            fp = plf.Fingerprint(vicinity_cutoff=3.0)
            fp.run_from_iterable([ligand_mol], protein_mol, progress=False)
            df = fp.to_dataframe().T
            hs_interactions = 0
            for inter, is_on in zip(df[0].index.values, df[0].values):
                if inter[-1] in ["HBDonor", "HBAcceptor"]:
                    hs_interactions += is_on
            interactions.append((ligand_mol.GetProp("_Name"), hs_interactions))
        except BaseException:
            pass
    return interactions


if __name__ == "__main__":
    import sys

    interactions = get_protein_ligand_interactions(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], "w") as csv:
        csv.write("mol,interactions\n")
        for mid, inter in interactions:
            csv.write(f"{mid},{inter}\n")
