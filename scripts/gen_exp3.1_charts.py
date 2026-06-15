#!/usr/bin/env python3
"""Generate Exp 3.1 charts from Phase 1 statistics results."""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = Path("/root/ESField/results/exp3.1_phase1_stats")
out_dir = results_dir / "charts"
out_dir.mkdir(parents=True, exist_ok=True)

# Load all data
pockets = []
success_rates = []
per_site_data = {}

for pdir in sorted(results_dir.iterdir()):
    if not pdir.is_dir() or pdir.name == "charts":
        continue
    f = pdir / "phase1_stats.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    pockets.append(pdir.name)
    success_rates.append(d["summary"]["success_rate"] * 100)
    per_site_data[pdir.name] = {
        int(k): v["coverage_rate"] for k, v in d["per_site_coverage"].items()
    }

# ── Chart 1: Bar chart ──
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2ca02c" if r > 0 else "#d62728" for r in success_rates]
bars = ax.bar(pockets, success_rates, color=colors, edgecolor="black", linewidth=0.8)
for bar, rate in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{rate:.0f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Anchor Success Rate (%)", fontsize=13)
ax.set_title("Phase 1 Anchor Success Rate (λ=5.0, 100 runs/pocket)", fontsize=14)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "fig_phase1_success_bar.png", dpi=150)
plt.close(fig)
print("✓ Bar chart saved")

# ── Chart 2: Heatmap ──
all_sites = sorted(set().union(*[set(d.keys()) for d in per_site_data.values()]))
n_pockets = len(pockets)
n_sites = len(all_sites)
heatmap = np.zeros((n_pockets, n_sites))
for i, p in enumerate(pockets):
    for j, s in enumerate(all_sites):
        heatmap[i, j] = per_site_data[p].get(s, 0.0) * 100

fig, ax = plt.subplots(figsize=(max(12, n_sites * 0.9), max(5, n_pockets * 0.8)))
im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(n_sites))
ax.set_xticklabels([f"Site {s}" for s in all_sites], fontsize=9)
ax.set_yticks(range(n_pockets))
ax.set_yticklabels(pockets, fontsize=10)
for i in range(n_pockets):
    for j in range(n_sites):
        val = heatmap[i, j]
        text_color = "white" if val > 50 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9,
                color=text_color, fontweight="bold" if val > 0 else "normal")
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Coverage Rate (%)", fontsize=11)
ax.set_title("Per-HEW-Site Anchor Coverage (100 Phase 1 runs/pocket)", fontsize=14)
fig.tight_layout()
fig.savefig(out_dir / "fig_phase1_coverage_heatmap.png", dpi=150)
plt.close(fig)
print("✓ Heatmap saved")

# ── Chart 3: Anchor type distribution ──
type_data = {}
for pdir in sorted(results_dir.iterdir()):
    if not pdir.is_dir() or pdir.name == "charts":
        continue
    f = pdir / "phase1_stats.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    type_data[pdir.name] = d.get("anchor_type_distribution", {})

all_types = set()
for td in type_data.values():
    all_types.update(td.keys())
all_types = sorted(all_types)

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(pockets))
for atype in all_types:
    counts = [type_data.get(p, {}).get(atype, {}).get("count", 0) for p in pockets]
    ax.bar(pockets, counts, bottom=bottom, label=atype)
    bottom += np.array(counts)
ax.set_ylabel("Total Anchor Count", fontsize=12)
ax.set_title("Anchor Atom Type Distribution per Pocket", fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(out_dir / "fig_phase1_anchor_types.png", dpi=150)
plt.close(fig)
print("✓ Anchor type chart saved")

print(f"\nAll charts saved to {out_dir}/")
