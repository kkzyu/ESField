from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

pdbs = Path(sys.argv[1]).glob("*")
affinities = {}
for pdb in pdbs:
    affinities[pdb.name] = {}
    for step in pdb.glob("*"):
        for score in step.glob("mols_vina/*score.txt"):
            try:
                a = open(score).readlines()[0].split("Affinity: ")[1].split("(")[0]
                a = float(a)
                lig_name = open(
                    score.parent / score.name.replace("_score.txt", ".pdbqt")
                ).readlines()[0]
                lig_name = lig_name.split("= ")[1].strip()
                if lig_name not in affinities[pdb.name]:
                    affinities[pdb.name][lig_name] = {}
                affinities[pdb.name][lig_name][step.name] = a
                # print(lig_name, a)
            except Exception:
                pass
        # if len(affinities) > 0:
        # if step.name not in steps:
        # steps[step.name] = []
        # steps[step.name] += affinities
        # print(pdb.name, step.name, f"{sum(affinities)/len(affinities):.3f}")

steps = {}

keys = set(["0", "100", "250", "500"])
# keys = set(["0", "100"]) #, "250", "500"])
# keys = set(["0", "100", "250"]) #, "500"])


for k in keys:
    steps[k] = []

for k, v in affinities.items():
    for kl, vl in v.items():
        if len(set(vl.keys()) & keys) == len(keys):
            for ks in keys:
                steps[ks].append(vl[ks])

print("MEAN")
for step in sorted(list(keys)):
    steps[step] = np.array(steps[step])
    positive_scores = np.sum(steps[step] <= 0) / len(steps[step])

    print(step, len(steps[step]), np.mean(steps[step]), positive_scores)

fig = plt.figure()
ax = fig.gca()
for step in sorted(list(keys)):
    x = steps[step].copy()
    x.sort()
    plt.plot(x, np.arange(len(x)) / (len(x) - 1), label=f"Guidance steps: {step}")
    # sns.kdeplot(, ax=ax, label=step)
plt.legend()
plt.xlim(x[0] - 0.5, 4)
plt.plot([0, 0], [0, 1], "--", linewidth=3, c="C3")
plt.savefig("vina.pdf", dpi=500, format="pdf")
plt.close()
