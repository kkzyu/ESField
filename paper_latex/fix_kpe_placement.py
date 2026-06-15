#!/usr/bin/env python3
"""Move KPE from 2.2 to 2.3. Fixed indentation matching."""
TEX = '/root/ESField/paper_latex/main.tex'
with open(TEX, 'r') as f:
    content = f.read()

# ── Edit A: Remove KPE from 2.2 ──
old_22 = (
    'decoupled from the generative process.  The \\textbf{Kinetic Path Energy\n'
    '(KPE) framework}~\\cite{li2026kpe} was formalised to quantify flow-matching\n'
    'transport costs.  While prior work uses KPE as a global trajectory\n'
    'diagnostic, we repurpose it to rigorously decompose and isolate the\n'
    'kinetic contribution of inference-time guidance signals ($E_{\\text{guide}}$)\n'
    'from the physiological ODE transport ($E_{\\text{ODE}}$), providing the\n'
    'energy, providing the quantitative foundation for our zero-strain\n'
    'alternative (Section~\\ref{sec:kpe_diagnosis}).'
)
new_22 = (
    'decoupled from the generative process.  This fundamental disconnect\n'
    'underscores the need for a differentiable, inference-time mechanism that\n'
    'can translate these thermodynamic priors into kinematic constraints\n'
    'without requiring model retraining.'
)
if old_22 in content:
    content = content.replace(old_22, new_22)
    print('Edit A: SUCCESS')
else:
    print('Edit A: FAIL')
    idx = content.find('decoupled from the generative process.  The')
    print('  Context:', repr(content[idx:idx+450]))

# ── Edit B: Insert KPE into 2.3 ──
old_23 = (
    'provides a mathematical zero-strain guarantee for site-specific constraints.\n'
    'Our kinematic anchor guidance addresses both gaps: site-type-level control\n'
    'and mathematically guaranteed kinematic consistency, as quantified by\n'
    'Table~\\ref{tab:method_comparison}.'
)
new_23 = (
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
if old_23 in content:
    content = content.replace(old_23, new_23)
    print('Edit B: SUCCESS')
else:
    print('Edit B: FAIL')
    idx = content.find('provides a mathematical zero-strain guarantee')
    print('  Context:', repr(content[idx:idx+250]))

with open(TEX, 'w') as f:
    f.write(content)
print('Done.')
