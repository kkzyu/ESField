"""Generate placeholder figures for the kinematic paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# Figure A: KPE Trajectory (4 panels)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
t = np.linspace(0, 1, 100)

# Panel (a): KPE accumulation
axes[0, 0].plot(t, 1.03 * t, 'k-', label='Baseline', linewidth=2)
spike_times = [0.3, 0.42, 0.55, 0.68, 0.80]
hard_kpe = 1.14 * t.copy()
for st in spike_times:
    idx = np.argmin(np.abs(t - st))
    hard_kpe[idx:] += 15.0
axes[0, 0].plot(t, hard_kpe, 'r-', label='Hard-Fix', linewidth=1.5)
axes[0, 0].plot(t, 1.03 * t + 0.0024 * t, 'b-', label='Kinematic (Ours)', linewidth=2)
axes[0, 0].set_xlabel('Integration Step t')
axes[0, 0].set_ylabel('Cumulative KPE')
axes[0, 0].set_title('(a) KPE Accumulation along ODE Trajectory')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel (b): Instantaneous velocity
v_baseline = 1.0 + 0.3 * np.sin(t * 20)
axes[0, 1].semilogy(t, v_baseline, 'k-', label='Baseline', linewidth=2)
v_hard = v_baseline.copy()
for spike_t in spike_times:
    idx = np.argmin(np.abs(t - spike_t))
    v_hard[idx] = 1200
axes[0, 1].semilogy(t, v_hard, 'r-', label='Hard-Fix', linewidth=1.5, alpha=0.7)
axes[0, 1].semilogy(t, v_baseline * 1.05, 'b-', label='Kinematic (Ours)', linewidth=2)
axes[0, 1].set_xlabel('Integration Step t')
axes[0, 1].set_ylabel('||v_eff|| (log scale)')
axes[0, 1].set_title('(b) Instantaneous Velocity Norm')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Panel (c): KPE ratio
axes[1, 0].axhline(y=0.985, color='r', linestyle='-', linewidth=2, label='Hard-Fix (98.5%)')
axes[1, 0].axhline(y=0.00006, color='b', linestyle='-', linewidth=2, label='Kinematic (0.006%)')
axes[1, 0].set_xlabel('Integration Step t')
axes[1, 0].set_ylabel('KPE Ratio')
axes[1, 0].set_title('(c) Cumulative KPE Ratio')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(-0.05, 1.1)

# Panel (d): Violin distributions
np.random.seed(42)
hard_kpe_dist = 0.985 + 0.01 * np.random.randn(50)
kinematic_kpe_dist = 0.00006 + 0.00002 * np.random.randn(50)
baseline_kpe_dist = np.zeros(50)
positions = [1, 2, 3]
axes[1, 1].violinplot([baseline_kpe_dist, hard_kpe_dist, kinematic_kpe_dist],
                       positions=positions, showmeans=True, showmedians=True)
axes[1, 1].set_xticks(positions)
axes[1, 1].set_xticklabels(['Baseline', 'Hard-Fix', 'Kinematic (Ours)'])
axes[1, 1].set_ylabel('Per-Molecule KPE Ratio')
axes[1, 1].set_title('(d) Per-Molecule KPE Ratio Distribution (N=50)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Figure A: Kinetic Path Energy (KPE) Trajectory Diagnostics',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/202605/WFMG/ESField/paper_latex/figures/figA_KPE_trajectory.pdf',
            dpi=150, bbox_inches='tight')
plt.close()
print('Figure A created.')

# ============================================================
# Figure B: Four-quadrant scatter + box plots
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Four-quadrant scatter
pockets = ['3mfw', '2gni', '6o4x', '2jke', '2gqn', '6phx']
delta_vina = [-1.2, -1.1, -0.3, -0.6, -0.4, -0.3]
delta_occ = [4, 4, 4, 4, 4, 4]
colors_list = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
np.random.seed(123)
for i, p in enumerate(pockets):
    ax1.scatter(delta_vina[i] + 0.05 * np.random.randn(),
                delta_occ[i] + 0.3 * np.random.randn(),
                c=colors_list[i], s=200, label=p,
                edgecolors='black', linewidth=1.5, zorder=5)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.fill_between([-3, 0], [0, 0], [8, 8], alpha=0.05, color='red')
ax1.fill_between([0, 3], [0, 0], [8, 8], alpha=0.12, color='green',
                 label='Pareto-dominant\n(better occupancy + affinity)')
ax1.set_xlabel('Delta Vina (Kinematic minus Hard-Fix) [kcal/mol]\n'
               '<-- better affinity',
               fontsize=11)
ax1.set_ylabel('Delta DirectOcc (Kinematic minus Hard-Fix) [%]\n'
               '--> better occupancy',
               fontsize=11)
ax1.set_title('(a) Four-Quadrant Breakthrough Scatter', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-2, 8)

# Panel (b): Box plots
box_data = {
    '3mfw': [np.random.normal(-6.2, 0.58, 50),
             np.random.normal(-5.6, 0.52, 50),
             np.random.normal(-6.8, 0.27, 50)],
    '2gni': [np.random.normal(-6.6, 0.55, 50),
             np.random.normal(-6.3, 0.49, 50),
             np.random.normal(-7.4, 0.24, 50)],
    '6o4x': [np.random.normal(-6.9, 0.50, 50),
             np.random.normal(-7.2, 0.46, 50),
             np.random.normal(-7.5, 0.23, 50)],
}
box_colors = ['gray', 'red', '#2196F3']
positions_list = [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
labels = ['3mfw\n(7 HEW)', '2gni\n(3 HEW)', '6o4x\n(6 HEW)']

for pi, (pocket, data) in enumerate(box_data.items()):
    for ci in range(3):
        bp = ax2.boxplot([data[ci]], positions=[positions_list[pi][ci]],
                         widths=0.55, patch_artist=True, showfliers=True,
                         showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='white',
                                       markersize=5))
        bp['boxes'][0].set_facecolor(box_colors[ci])
        bp['boxes'][0].set_alpha(0.7)

ax2.set_xticks([2, 6, 10])
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel('Vina Score [kcal/mol]', fontsize=11)
ax2.set_title('(b) Binding Affinity: Per-Pocket Box Plots', fontsize=12,
              fontweight='bold')
ax2.axhline(y=-7.0, color='green', linestyle=':', alpha=0.4,
            label='Strong binder')
legend_elements = [Patch(facecolor='gray', alpha=0.7, label='Baseline'),
                   Patch(facecolor='red', alpha=0.7, label='Hard-Fix (v7.1)'),
                   Patch(facecolor='#2196F3', alpha=0.7,
                         label='Kinematic (Ours)')]
ax2.legend(handles=legend_elements, fontsize=9, loc='lower left')
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure B: Occupancy-Affinity Paradox Resolution',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/202605/WFMG/ESField/paper_latex/figures/figB_quadrant_boxplot.pdf',
            dpi=150, bbox_inches='tight')
plt.close()
print('Figure B created.')

# ============================================================
# Figure C: 3D conformational comparison (schematic)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

# Left: Worst hard-fix molecule
ax1.set_xlim(0, 12); ax1.set_ylim(0, 10)
protein = plt.Circle((4, 5), 3.0, fill=True, color='lightgray',
                      alpha=0.4, ec='gray', linewidth=1.5)
ax1.add_patch(protein)
hew = plt.Circle((8.5, 5), 0.35, fill=True, color='blue', alpha=0.8)
ax1.add_patch(hew)
ax1.text(8.5, 5.7, 'HEW Site', ha='center', fontsize=9, color='darkblue',
         fontweight='bold')
# Anchor atoms (rigidly fixed)
anchors_x = [8.2, 8.8, 8.5, 8.3]
anchors_y = [4.6, 5.3, 5.15, 4.85]
ax1.scatter(anchors_x, anchors_y, c='cyan', s=150, edgecolors='black',
            linewidth=1.5, zorder=5)
ax1.text(8.5, 4.0, 'Rigidly Fixed\nAnchors', ha='center', fontsize=9,
         color='darkcyan', fontweight='bold')
# Steric clash zones
clash_pts = [(3.0, 3.0), (1.8, 5.5), (2.8, 7.0), (5.5, 3.2), (5.2, 7.2)]
for cx, cy in clash_pts:
    ax1.add_patch(plt.Circle((cx, cy), 0.55, fill=False, color='red',
                             linewidth=2.5, linestyle='--'))
    ax1.scatter([cx], [cy], c='red', s=80, marker='x', linewidth=2)
ax1.text(3.5, 1.5, 'Steric Clashes\n(red circles)', ha='center', fontsize=9,
         color='darkred', fontweight='bold')
# Ligand atoms (strained)
ax1.scatter([6.5, 7.0, 6.0, 5.5, 6.8], [3.5, 6.5, 6.0, 3.8, 6.8],
            c='orange', s=80, alpha=0.6, edgecolors='darkorange', linewidth=1)
ax1.set_title('Hard-Fix (v7.1): Conformational Collapse\n'
              'Vina = -1.70 kcal/mol  |  KPE Ratio = 98.5%',
              fontsize=12, fontweight='bold', color='darkred', pad=25)
# Legend
from matplotlib.lines import Line2D
legend1 = [Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
                  markersize=10, markeredgecolor='black', label='Anchor atoms'),
           Line2D([0], [0], marker='x', color='red', markersize=10,
                  markeredgewidth=2, linestyle='None', label='Steric clash')]
ax1.legend(handles=legend1, fontsize=8, loc='upper right')
ax1.set_aspect('equal')
ax1.axis('off')

# Right: Best kinematic molecule
ax2.set_xlim(0, 12); ax2.set_ylim(0, 10)
protein2 = plt.Circle((4, 5), 3.0, fill=True, color='lightgray',
                       alpha=0.4, ec='gray', linewidth=1.5)
ax2.add_patch(protein2)
hew2 = plt.Circle((8.5, 5), 0.35, fill=True, color='blue', alpha=0.8)
ax2.add_patch(hew2)
ax2.text(8.5, 5.7, 'HEW Site', ha='center', fontsize=9, color='darkblue',
         fontweight='bold')
# Relaxed anchor atoms
ax2.scatter([8.2, 8.8, 8.5, 8.3], [4.6, 5.3, 5.15, 4.85],
            c='green', s=150, edgecolors='black', linewidth=1.5, zorder=5)
ax2.text(8.5, 4.0, 'Relaxed\nAnchors', ha='center', fontsize=9,
         color='darkgreen', fontweight='bold')
# Full relaxed ligand
theta = np.linspace(0, 2*np.pi, 14)
lig_x = 8.5 + 1.8 * np.cos(theta)
lig_y = 5.0 + 1.8 * np.sin(theta)
ax2.scatter(lig_x, lig_y, c='#FFD700', s=90, alpha=0.8,
            edgecolors='#B8860B', linewidth=1.2)
# Connecting bonds (lines between nearby atoms)
for i in range(len(lig_x)):
    for j in range(i+1, len(lig_x)):
        d = np.sqrt((lig_x[i]-lig_x[j])**2 + (lig_y[i]-lig_y[j])**2)
        if d < 2.2:
            ax2.plot([lig_x[i], lig_x[j]], [lig_y[i], lig_y[j]],
                     '-', color='#B8860B', linewidth=0.8, alpha=0.5)
# No clash indicators
ax2.text(2.5, 8.5, 'No steric clashes', fontsize=10, color='darkgreen',
         fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax2.set_title('Kinematic (Ours): Global Relaxation\n'
              'Vina = -8.2 kcal/mol  |  KPE Ratio = 0.006%',
              fontsize=12, fontweight='bold', color='darkgreen', pad=25)
legend2 = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=10, markeredgecolor='black',
                  label='Relaxed anchors'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
                  markersize=8, markeredgecolor='#B8860B',
                  label='Ligand scaffold')]
ax2.legend(handles=legend2, fontsize=8, loc='upper right')
ax2.set_aspect('equal')
ax2.axis('off')

plt.suptitle('Figure C: 3D Conformational Comparison — '
             'Spatial Clash vs. Global Relaxation (3mfw Pocket)',
             fontsize=14, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.85)
plt.savefig('F:/202605/WFMG/ESField/paper_latex/figures/figC_3D_comparison.pdf',
            dpi=150, bbox_inches='tight')
plt.savefig('F:/202605/WFMG/ESField/paper_latex/figures/figC_3D_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Figure C created.')
print('\nAll 3 placeholder figures generated successfully!')
