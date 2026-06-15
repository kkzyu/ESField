from src.guidance_plugins import forcefield
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

energies = []
starts = []
betas = []
natoms = []
nmols = []
files = os.listdir("raw_mols")
for file in tqdm(files):
    try:
        if "trajectory" not in file:
            filel = file.split("_")
            beta = int(filel[2].split(".")[0])
            start = int(filel[3])
            nmol = int(filel[0])
            natom = int(filel[1])
            samples = np.load("raw_mols/" + file)
            energy_batch, _ = forcefield.score(
                samples[..., :3], samples[..., 3, np.newaxis], requires_grad=False
            )
            energies.append(np.nanmean(energy_batch))
            starts.append(start)
            natoms.append(natom)
            betas.append(beta)
            nmols.append(nmol)
    except Exception:
        pass


df = pd.DataFrame(
    {"start": starts, "beta": betas, "natom": natoms, "nmol": nmols, "energy": energies}
)
df.to_csv("results.csv")
