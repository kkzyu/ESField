#!/usr/bin/env python3
"""Major restructure: Abstract + Intro pipeline overview + RW gap + scoped zero-strain.
Run: python3 /root/ESField/paper_latex/restructure_abstract_intro_rw.py
"""
import re

TEX = '/root/ESField/paper_latex/main.tex'
with open(TEX, 'r') as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════
# EDIT 1: Abstract — restructure with E_site + pipeline + scoped zero-strain
# ═══════════════════════════════════════════════════════════════
abs_start = content.find(r'\begin{abstract}')
abs_end = content.find(r'\end{abstract}') + len(r'\end{abstract}')

new_abstract = r'''\begin{abstract}
A grand challenge in structure-based 3D molecular generation is targeting
high-energy water (HEW) sites to improve binding affinity, yet existing
generators suffer from \textbf{water-blindness}---they learn no
physicochemical prior for water displacement.  We address this by first
identifying HEW sites and constructing a differentiable compatibility
energy $E_{\text{site}}$ that attracts chemically compatible atoms.
During molecule growth, instead of applying the gradient of
$E_{\text{site}}$ directly to individual atom coordinates, we
orthogonally decompose the guidance displacement field and project it
\emph{exclusively} onto the centre-of-mass (CoM) subspace.  This
guarantees that the guidance update itself injects \textbf{zero strain}
into all internal coordinates (bond lengths, angles, torsions)---the
molecule is softly pulled toward the target as a rigid body, inherently
preventing steric clashes.  In stark contrast, naive coordinate fixation
(``hard-fix'') injects a \textbf{31,468-fold surge} in destructive
kinetic energy (98.5\% of total energy), directly causing severe
conformational strain and degraded docking scores.  Across
cross-architecture validation on the flow-matching-based DrugFlow and the
DDPM-based TargetDiff---two structurally distinct generative
frameworks---on 6 pharmacologically diverse PDBbind pockets, kinematic
anchoring delivers: (i)~\textbf{31,468$\times$ KPE suppression}
($\KPERatio = 0.006\%$ vs.\ 98.5\% for hard-fix), providing ironclad
physical validation of the zero-strain guarantee;
(ii)~\textbf{simultaneous Pareto-improvement in both occupancy and
affinity}---the best Vina score in every pocket with Wilcoxon
signed-rank $p = 4 \times 10^{-6}$ over hard-fix; and
(iii)~\textbf{2$\times$ variance reduction} ($\sigma_{\text{Vina}}$:
$0.52 \to 0.27$), entirely eliminating the catastrophic conformational
collapse (Vina $=-1.70$~kcal/mol) and repulsive outliers
($+6.86$~kcal/mol) that plague hard-fix.  These results establish that
physically consistent, zero-strain hydration-site targeting is both
achievable and essential for the next generation of water-aware molecular
design.
\end{abstract}'''

content = content[:abs_start] + new_abstract + content[abs_end:]

# ═══════════════════════════════════════════════════════════════
# EDIT 2: Intro — shorten hard-fix para, add pipeline overview
# ═══════════════════════════════════════════════════════════════
intro_start = content.find(r'\section{Introduction}')
contrib_marker = r'Our contributions are:'
intro_contrib_start = content.find(contrib_marker)
intro_end = content.find(r'\section{Related Work}')

# Keep Para 1 as-is (water-blindness)
para1_end_marker = 'objective of water displacement.'
para1_end = content.find(para1_end_marker, intro_start) + len(para1_end_marker)

para1 = content[intro_start:para1_end]

# New shortened Para 2 (hard-fix + KPE, 5-6 lines)
new_para2 = r'''

Faced with this blindness, the intuitive strategy is ``hard-fix'':
rigidly overwriting anchor-atom coordinates to enforce site
occupancy~\cite{griesbacher2026ebmol,esfield2026}.  Our Kinetic Path
Energy (KPE) diagnosis~\cite{li2026kpe} reveals this to be physically
catastrophic: each overwrite constitutes an instantaneous teleport,
injecting \textbf{98.5\% excess kinetic energy}---a 31,468-fold
surge---that locks internal conformational degrees of freedom and
produces the \textbf{Occupancy--Affinity Paradox} (anchored molecules
exhibit a detrimental Vina trend; e.g., Cliff's $\delta=+0.47$).
This is not an implementation flaw but a \textbf{fundamental
dimensionality mismatch}: hard-fix applies a
$\reals^{3N_{\text{atoms}}}$-dimensional correction to a $\reals^3$
control objective, turning excess degrees of freedom into channels for
unbounded kinetic energy injection.

'''

# New Para 3 (pipeline overview + method)
new_para3 = r'''To resolve this, we propose \textbf{Kinematic Anchor Guidance}, a
physically principled framework that proceeds in three stages.
\textbf{First}, we detect HEW sites and construct a differentiable
compatibility energy $E_{\text{site}}$ that attracts chemically compatible
atoms to target sites based on microenvironment type
(Section~\ref{sec:methods}).  \textbf{Second}, we adopt a two-stage
generation process: a small fragment is grown to occupy the target
HEW site (Phase~1, Occupy), then the full molecule is expanded around
these anchor atoms (Phase~2, Connect).  \textbf{Third}, during Phase~2,
instead of overwriting anchor coordinates, we orthogonally decompose the
gradient of $E_{\text{site}}$ and project it exclusively onto the
centre-of-mass (CoM) subspace ($\reals^3$).  This dimensionality-matched
intervention mathematically guarantees that the guidance update itself
injects \textbf{zero strain} on all internal coordinates (bond lengths,
angles, torsions; Theorem~1)---the molecule is softly pulled toward
the target as a rigid body, preserving full internal flexibility and
allowing the underlying generative process to govern conformational
relaxation without geometric interference.

'''

# Contributions keep the existing block
contrib_block = content[intro_contrib_start:intro_end]

# Assemble new intro
new_intro = para1 + new_para2 + new_para3 + '\n' + contrib_block

old_intro = content[intro_start:intro_end]
content = content.replace(old_intro, new_intro)
print('EDIT 2 (Intro restructure): SUCCESS')

# ═══════════════════════════════════════════════════════════════
# EDIT 3: Related Work 2.3 — add dimensionality mismatch gap
# ═══════════════════════════════════════════════════════════════
old_rw_gap = (
    'neither\n'
    'provides a mathematical zero-strain guarantee for site-specific constraints.\n'
    'The \\textbf{Kinetic Path Energy (KPE) framework}~\\cite{li2026kpe},\n'
    'originally formalised to quantify global flow-matching transport costs,\n'
    'has not previously been leveraged to isolate the kinetic contribution of\n'
    'inference-time guidance signals.  We repurpose this framework to\n'
    'rigorously decompose guidance-induced kinetic energy ($E_{\\text{guide}}$)\n'
    'from physiological ODE transport ($E_{\\text{ODE}}$), providing the\n'
    'quantitative foundation that exposes the catastrophic physical cost of\n'
    'naive coordinate fixation (Section~\\ref{sec:kpe_diagnosis}).\n'
    'Our kinematic anchor guidance addresses both gaps: site-type-level control\n'
    'and mathematically guaranteed kinematic consistency, as quantified by\n'
    'Table~\\ref{tab:method_comparison}.'
)

new_rw_gap = (
    'neither\n'
    'provides a mathematical zero-strain guarantee for site-specific constraints\n'
    'under a kinematic decomposition.  Crucially, site-specific constraints\n'
    '(such as HEW targeting) represent a \\textbf{dimensionality mismatch}: a\n'
    'local $\\reals^3$ objective enforced on a $\\reals^{3N}$ molecular\n'
    'manifold.  Existing guidance methods lack a principled way to resolve this\n'
    'mismatch; they either apply global force fields or rigidly overwrite\n'
    'coordinates, both of which inject uncontrolled kinetic energy into internal\n'
    'degrees of freedom.  The \\textbf{Kinetic Path Energy (KPE)\n'
    'framework}~\\cite{li2026kpe}, originally formalised to quantify global\n'
    'flow-matching transport costs, has not previously been leveraged to\n'
    'isolate the kinetic contribution of inference-time guidance signals.  We\n'
    'repurpose this framework to rigorously decompose guidance-induced kinetic\n'
    'energy ($E_{\\text{guide}}$) from physiological ODE transport\n'
    '($E_{\\text{ODE}}$), providing the quantitative foundation that exposes\n'
    'the catastrophic physical cost of naive coordinate fixation\n'
    '(Section~\\ref{sec:kpe_diagnosis}).\n'
    'Our kinematic anchor guidance addresses both gaps: site-type-level control\n'
    'and mathematically guaranteed kinematic consistency, as quantified by\n'
    'Table~\\ref{tab:method_comparison}.'
)

if old_rw_gap in content:
    content = content.replace(old_rw_gap, new_rw_gap)
    print('EDIT 3 (RW 2.3 dimensionality mismatch): SUCCESS')
else:
    print('EDIT 3: NOT FOUND')
    idx = content.find('provides a mathematical zero-strain guarantee for site-specific constraints')
    if idx >= 0:
        print('  Context:', repr(content[idx:idx+400]))

# ═══════════════════════════════════════════════════════════════
# EDIT 4: Global — scope "zero strain" to guidance update
# ═══════════════════════════════════════════════════════════════
replacements = [
    # Abstract
    ('mathematically guarantees \\textbf{zero strain} on all internal coordinates',
     'mathematically guarantees that the guidance update itself injects \\textbf{zero strain} into all internal coordinates'),
    # Abstract: "ironclad physical validation of the zero-strain guarantee"
    # keep as-is since it refers to the guarantee already scoped

    # Intro contributions
    ('our method provides a mathematical zero-strain\n\t    guarantee (Theorem~1)',
     'our method provides a mathematical guarantee of\n\t    zero strain in the guidance update (Theorem~1)'),

    # Intro "mathematically guarantees zero strain on all internal coordinates"
    ('mathematically guarantees that the guidance update itself\ninjects \\textbf{zero strain} on all internal coordinates (bond lengths,',
     'mathematically guarantees that the guidance update itself\ninjects \\textbf{zero strain} into all internal coordinates (bond lengths,'),

    # Methods Theorem statement
    ('\\emph{Zero-Strain Guarantee',
     '\\emph{Zero-Strain Guidance Guarantee'),
]

for old_s, new_s in replacements:
    if old_s in content:
        content = content.replace(old_s, new_s)
        print(f'  Replaced: {old_s[:60]}...')
    else:
        print(f'  NOT FOUND: {old_s[:60]}...')

# ═══════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════
with open(TEX, 'w') as f:
    f.write(content)

print('\nAll edits applied.')
print('Verification checklist:')
print('  [ ] Abstract: HEW + E_site + CoM projection + zero-strain (guidance step) in first 3 sentences?')
print('  [ ] Abstract: hard-fix contrast <= 2 sentences?')
print('  [ ] Intro: Pipeline Overview (3 stages) before contributions?')
print('  [ ] Intro: E_site source mentioned (site detection + compatibility)?')
print('  [ ] RW 2.3: dimensionality mismatch explicitly mentioned?')
print('  [ ] Global: zero strain scoped to guidance update?')
print('  [ ] Abstract: "and annealing baselines" removed?')
