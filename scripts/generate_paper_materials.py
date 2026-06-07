#!/usr/bin/env python3
"""Generate paper materials from v7.1 ablation study results.

Produces:
  - LaTeX tables (generalization, ablation summary, failure analysis)
  - Figures (DirectOcc bar plot, ablation heatmap)
  - Statistical significance matrix (CSV)

Usage:
    PYTHONPATH=src python scripts/generate_paper_materials.py
"""

import json, os, sys, csv
from pathlib import Path
from math import comb as binom_coeff
import numpy as np

ESFIELD_ROOT = "/root/ESField"
ABLATION_JSON = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/v71_ablation/ablation_results.json"
OUTPUT_DIR = f"{ESFIELD_ROOT}/paper"
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

# ── Data loading ──
data = json.load(open(ABLATION_JSON))
pockets = data["pockets"]
conditions = [c["name"] for c in data["conditions"]]
results = data["results"]

# ── Helper: Clopper-Pearson CI ──
def cp_ci(k, n, alpha=0.05):
    from scipy.stats import beta as beta_dist
    if n == 0: return (0.0, 1.0)
    lo = beta_dist.ppf(alpha/2, k, n-k+1) if k > 0 else 0.0
    hi = beta_dist.ppf(1-alpha/2, k+1, n-k) if k < n else 1.0
    return (float(lo), float(hi))

def binom_pval(k, n, p0=0.0):
    if p0 == 0: return 1e-10 if k > 0 else 1.0
    return min(sum(binom_coeff(n,i)*(p0**i)*((1-p0)**(n-i)) for i in range(k,n+1)), 1.0)

# Extract per-pocket-condition metrics
def get_metric(pocket, cond, key, default=0.0):
    pr = results.get(pocket, {}).get(cond, {})
    return pr.get(key, default)

# ===================================================================
# TABLE 1: Generalization Performance (v7.1_full only, per pocket)
# ===================================================================
print("=" * 70)
print("TABLE 1: Generalization Performance (v7.1_full)")
print("=" * 70)

table1_rows = []
for p in pockets:
    n_v = get_metric(p, "v7.1_full", "n_valid", 0)
    n_occ = get_metric(p, "v7.1_full", "n_occupied", 0)
    rate = get_metric(p, "v7.1_full", "direct_occ_rate", 0.0)
    ci = cp_ci(n_occ, n_v) if n_v > 0 else (0, 0)
    pval = binom_pval(n_occ, n_v) if n_v > 0 else 1.0
    qed_m = get_metric(p, "v7.1_full", "qed_mean", 0.0)
    qed_s = get_metric(p, "v7.1_full", "qed_std", 0.0)
    posu_m = get_metric(p, "v7.1_full", "posu_mean", 0.0)
    posu_s = get_metric(p, "v7.1_full", "posu_std", 0.0)
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
    table1_rows.append((p, n_v, n_occ, rate, ci, pval, sig, qed_m, qed_s, posu_m, posu_s))

# LaTeX
latex1 = r"""\begin{table}[t]
\centering
\caption{Generalization performance of v7.1 across 10 PDBbind pockets.
DirectOcc = fraction of generated molecules with at least one compatible
atom within 2.5\,\AA\ of a candidate HEW site. Baseline (v6-D.2) DirectOcc
= 0\% on all pockets. p-values from one-sided binomial test vs.\ $p_0=0$.}
\label{tab:generalization}
\begin{tabular}{lrrrrcrr}
\toprule
Pocket & $N$ & Occupied & DirectOcc (\%) & 95\% CI & $p$ & QED & POSU \\
\midrule
"""
for p, n_v, n_occ, rate, ci, pval, sig, qm, qs, pm, ps in table1_rows:
    if n_v == 0:
        latex1 += f"{p} & — & — & — & — & — & — & — \\\\\n"
        continue
    latex1 += (f"{p} & {n_v} & {n_occ} & {rate*100:.1f} & "
               f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}] & "
               f"{pval:.1e}" + f"$^{{{{{sig}}}}}$" + " & "
               f"${qm:.2f}\\pm{qs:.2f}$ & ${pm:.3f}\\pm{ps:.3f}$ \\\\\n")

latex1 += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex1)
with open(f"{OUTPUT_DIR}/tables/table1_generalization.tex", "w") as f:
    f.write(latex1)

# ===================================================================
# TABLE 2: Ablation Summary (mean ± std across successful pockets)
# ===================================================================
print("\n" + "=" * 70)
print("TABLE 2: Ablation Study Summary")
print("=" * 70)

# Successful pockets: those where v7.1_full had n_valid > 0
ok_pockets = [p for p in pockets if get_metric(p, "v7.1_full", "n_valid", 0) > 0]
print(f"Successful pockets for Table 2: {ok_pockets}")

table2_data = {}
for cond in conditions:
    rates, qeds, posus = [], [], []
    for p in ok_pockets:
        n_v = get_metric(p, cond, "n_valid", 0)
        if n_v > 0:
            rates.append(get_metric(p, cond, "direct_occ_rate", 0))
            qeds.append(get_metric(p, cond, "qed_mean", 0))
            posus.append(get_metric(p, cond, "posu_mean", 0))
    table2_data[cond] = {
        "n_ok": len(rates),
        "occ_mean": np.mean(rates) if rates else 0,
        "occ_std": np.std(rates) if rates else 0,
        "qed_mean": np.mean(qeds) if qeds else 0,
        "qed_std": np.std(qeds) if qeds else 0,
        "posu_mean": np.mean(posus) if posus else 0,
        "posu_std": np.std(posus) if posus else 0,
    }

latex2 = r"""\begin{table}[t]
\centering
\caption{Ablation study: mean $\pm$ std across %
"""
latex2 += f"{len(ok_pockets)} pockets where Phase~1 succeeded. "
latex2 += r"""Each condition: 25 molecules per pocket.}
\label{tab:ablation}
\begin{tabular}{lrrrr}
\toprule
Condition & DirectOcc (\%) & QED & POSU & $N_{\text{pockets}}$ \\
\midrule
"""
for cond in conditions:
    d = table2_data[cond]
    latex2 += (f"{cond.replace('_', '\\_')} & "
               f"${d['occ_mean']*100:.1f}\\pm{d['occ_std']*100:.1f}$ & "
               f"${d['qed_mean']:.2f}\\pm{d['qed_std']:.2f}$ & "
               f"${d['posu_mean']:.3f}\\pm{d['posu_std']:.3f}$ & "
               f"{d['n_ok']} \\\\\n")
latex2 += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex2)
with open(f"{OUTPUT_DIR}/tables/table2_ablation.tex", "w") as f:
    f.write(latex2)

# ===================================================================
# TABLE 3: Failure Pocket Analysis
# ===================================================================
print("\n" + "=" * 70)
print("TABLE 3: Failure Pocket Characteristics")
print("=" * 70)

fail_pockets = ["1h0r", "3t09", "3fzn", "3f35"]
site_dir = f"{ESFIELD_ROOT}/experiments/pdbbind_water_sites/site_maps"

sys.path.insert(0, f"{ESFIELD_ROOT}/src")
from guidance.latent_guidance import classify_hew_environment

fail_rows = []
for p in fail_pockets:
    sm_path = os.path.join(site_dir, f"{p}_site_map.json")
    if not os.path.exists(sm_path):
        fail_rows.append((p, "N/A", "N/A", "N/A", "Site map not found"))
        continue
    sm = json.load(open(sm_path))
    hew = [s for s in sm["sites"] if s["site_type"] == "high_energy_water"]
    envs = {}
    min_prot_dists = []
    for s in hew:
        env = classify_hew_environment(s)
        envs[env] = envs.get(env, 0) + 1
        d = s.get("features", {}).get("nearest_protein_distance", 99)
        min_prot_dists.append(d)

    env_str = ", ".join(f"{e}:{c}" for e, c in sorted(envs.items()))
    min_d = min(min_prot_dists) if min_prot_dists else 99

    # Diagnose failure reason
    if min_d < 2.5:
        reason = "HEW sites buried (<2.5Å from protein); inaccessible to DrugFlow"
    elif min_d < 3.5:
        reason = "HEW sites in tight pockets; limited conformational space"
    else:
        reason = "Site geometry or pocket shape incompatible with DrugFlow fragment growth"

    # Phase 1 success?
    p1_ok = get_metric(p, "v7.1_full", "n_valid", 0) > 0
    if not p1_ok:
        reason += "; Phase 1 failed (no anchor found)"

    fail_rows.append((p, len(hew), f"{min_d:.1f}", env_str, reason))

latex3 = r"""\begin{table}[t]
\centering
\caption{Characteristics of pockets where v7.1 produced zero DirectOcc.
All metrics from the v7.1\_full condition.}
\label{tab:failures}
\begin{tabular}{lrrp{3cm}p{5cm}}
\toprule
Pocket & $N_{\text{HEW}}$ & Min. prot.\ dist.\ (\AA) & Env.\ dist. & Likely failure reason \\
\midrule
"""
for p, n_hew, min_d, env_str, reason in fail_rows:
    latex3 += f"{p} & {n_hew} & {min_d} & {env_str} & {reason} \\\\\n"
latex3 += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex3)
with open(f"{OUTPUT_DIR}/tables/table3_failures.tex", "w") as f:
    f.write(latex3)

# ===================================================================
# FIGURE 1: DirectOcc Bar Plot
# ===================================================================
print("\n" + "=" * 70)
print("FIGURE 1: DirectOcc Bar Plot")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(pockets))
width = 0.35

rates = [get_metric(p, "v7.1_full", "direct_occ_rate", 0) * 100 for p in pockets]
cis_lo = []
cis_hi = []
for p in pockets:
    n_v = get_metric(p, "v7.1_full", "n_valid", 0)
    n_occ = get_metric(p, "v7.1_full", "n_occupied", 0)
    lo, hi = cp_ci(n_occ, n_v) if n_v > 0 else (0, 0)
    cis_lo.append(max(0, (rates[pockets.index(p)]/100 - lo) * 100))
    cis_hi.append(max(0, (hi - rates[pockets.index(p)]/100) * 100))

errors = [cis_lo, cis_hi]
bars = ax.bar(x, rates, width, yerr=errors, capsize=4,
              color=["#2196F3" if r > 0 else "#FF5252" for r in rates],
              edgecolor="black", linewidth=0.5)

ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(y=10, color="green", linestyle=":", linewidth=0.8, label="10% threshold")

ax.set_xticks(x)
ax.set_xticklabels(pockets, rotation=45, ha="right")
ax.set_ylabel("Direct Occupancy Rate (%)")
ax.set_title("v7.1 DirectOcc Across 10 PDBbind Pockets\n(v6-D.2 baseline: 0% on all)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

# Add count labels
for i, (r, p) in enumerate(zip(rates, pockets)):
    n_occ = get_metric(p, "v7.1_full", "n_occupied", 0)
    n_v = get_metric(p, "v7.1_full", "n_valid", 0)
    if n_v > 0:
        ax.text(i, r + 2, f"{n_occ}/{n_v}", ha="center", fontsize=8)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figures/fig_direct_occ_barplot.png", dpi=300)
fig.savefig(f"{OUTPUT_DIR}/figures/fig_direct_occ_barplot.pdf")
plt.close()
print("Saved fig_direct_occ_barplot.png/pdf")

# ===================================================================
# FIGURE 2: Ablation Heatmap
# ===================================================================
print("\n" + "=" * 70)
print("FIGURE 2: Ablation Heatmap")
print("=" * 70)

# Build matrix: pockets (rows) × conditions (cols)
ok_ranked = sorted(ok_pockets, key=lambda p: -get_metric(p, "v7.1_full", "direct_occ_rate", 0))
matrix = np.zeros((len(ok_ranked), len(conditions)))
annot = [["" for _ in conditions] for _ in ok_ranked]

for i, p in enumerate(ok_ranked):
    for j, cond in enumerate(conditions):
        rate = get_metric(p, cond, "direct_occ_rate", 0)
        matrix[i, j] = rate * 100
        n_v = get_metric(p, cond, "n_valid", 0)
        if n_v == 0:
            annot[i][j] = "FAIL"
        else:
            annot[i][j] = f"{rate*100:.0f}%"

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(40, matrix.max()))

ax.set_xticks(range(len(conditions)))
ax.set_xticklabels([c.replace("_", "\n") for c in conditions], rotation=45, ha="right")
ax.set_yticks(range(len(ok_ranked)))
ax.set_yticklabels(ok_ranked)

for i in range(len(ok_ranked)):
    for j in range(len(conditions)):
        color = "white" if matrix[i, j] > 25 else "black"
        ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=9, color=color)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("DirectOcc (%)")
ax.set_title("v7.1 Ablation Study: DirectOcc Heatmap")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figures/fig_ablation_heatmap.png", dpi=300)
fig.savefig(f"{OUTPUT_DIR}/figures/fig_ablation_heatmap.pdf")
plt.close()
print("Saved fig_ablation_heatmap.png/pdf")

# ===================================================================
# Significance Matrix (CSV)
# ===================================================================
print("\n" + "=" * 70)
print("SIGNIFICANCE MATRIX")
print("=" * 70)

sig_csv = os.path.join(OUTPUT_DIR, "tables", "significance_matrix.csv")
with open(sig_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pocket"] + conditions)
    for p in pockets:
        row = [p]
        for cond in conditions:
            n_v = get_metric(p, cond, "n_valid", 0)
            n_occ = get_metric(p, cond, "n_occupied", 0)
            pval = binom_pval(n_occ, n_v) if n_v > 0 else 1.0
            if n_v == 0:
                row.append("FAIL")
            elif pval < 0.001:
                row.append("***")
            elif pval < 0.01:
                row.append("**")
            elif pval < 0.05:
                row.append("*")
            else:
                row.append("n.s.")
        writer.writerow(row)

# Print readable matrix
for p in pockets:
    parts = [f"{p:<8}"]
    for cond in conditions:
        n_v = get_metric(p, cond, "n_valid", 0)
        n_occ = get_metric(p, cond, "n_occupied", 0)
        pval = binom_pval(n_occ, n_v) if n_v > 0 else 1.0
        if n_v == 0:
            parts.append("FAIL".ljust(8))
        elif pval < 0.001:
            parts.append("***".ljust(8))
        elif pval < 0.01:
            parts.append("**".ljust(8))
        elif pval < 0.05:
            parts.append("*".ljust(8))
        else:
            parts.append(f"ns({pval:.2f})".ljust(8))
    print(" ".join(parts))

print(f"\nSignificance matrix saved to {sig_csv}")

# ===================================================================
# CONSOLIDATED OUTPUT SUMMARY
# ===================================================================
print(f"\n{'='*70}")
print("PAPER MATERIALS GENERATED")
print(f"{'='*70}")
print(f"Tables:")
print(f"  {OUTPUT_DIR}/tables/table1_generalization.tex")
print(f"  {OUTPUT_DIR}/tables/table2_ablation.tex")
print(f"  {OUTPUT_DIR}/tables/table3_failures.tex")
print(f"  {OUTPUT_DIR}/tables/significance_matrix.csv")
print(f"Figures:")
print(f"  {OUTPUT_DIR}/figures/fig_direct_occ_barplot.png")
print(f"  {OUTPUT_DIR}/figures/fig_direct_occ_barplot.pdf")
print(f"  {OUTPUT_DIR}/figures/fig_ablation_heatmap.png")
print(f"  {OUTPUT_DIR}/figures/fig_ablation_heatmap.pdf")
print(f"{'='*70}")
