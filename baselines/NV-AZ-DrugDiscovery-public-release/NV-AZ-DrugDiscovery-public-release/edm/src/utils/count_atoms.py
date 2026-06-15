import sys

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

x = Chem.SDMolSupplier(sys.argv[1], sanitize=False, removeHs=False)[0]
print(x.GetNumAtoms())
