#!/usr/bin/env python3
"""Compress Intro + RW based on 对话2 final recommendations.
Run: python3 /root/ESField/paper_latex/compress_intro_rw.py
"""
import re

TEX = '/root/ESField/paper_latex/main.tex'
with open(TEX, 'r') as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════
# COMPRESSION 1: Intro — merge 6 loose paragraphs into 3 tight ones
# ═══════════════════════════════════════════════════════════════

# Find Intro boundaries: from \section{Introduction} to \section{Related Work}
intro_start = content.find(r'\section{Introduction}')
rw_marker = r'\section{Related Work}'
intro_end = content.find(rw_marker)

old_intro_body = content[intro_start:intro_end]

# Find the Contributions block — keep it intact
contrib_start_marker = r'Our contributions are:'
contrib_idx = old_intro_body.find(contrib_start_marker)

# Everything before contributions is what we compress
body_before_contrib = old_intro_body[:contrib_idx]
contrib_block = old_intro_body[contrib_idx:]

# Build compressed intro
compressed_intro = r'''\section{Introduction}

Despite the well-established thermodynamic value of displacing high-energy
water (HEW, worth 0.5--2.0~kcal/mol in binding
affinity~\cite{dunitz1994,abel2008,michel2009,ladbury1996,spyrakis2017}),
recent structure-based 3D molecular
generators---TargetDiff~\cite{guan2023targetdiff},
DiffSBDD~\cite{schneuing2024diffsbdd}, DecompDiff~\cite{guan2023decompdiff},
DrugFlow~\cite{drugflow2024}, and FLAG~\cite{zhang2024flag}---suffer from
severe \textbf{water-blindness}.  Because Protein Data
Bank~\cite{berman2000pdb} training corpora rarely retain explicit
interfacial water molecules, these models learn a distribution
$p(\text{ligand} \mid \text{pocket})$ that fundamentally lacks the
physicochemical prior for water displacement: the model has no incentive
to replace an energetically unfavorable water molecule, and may equally
well place a methyl group adjacent to a HEW site, forfeiting a significant
binding affinity opportunity~\cite{abel2008,michel2009}.  Empirically,
this blindness is striking: across 10 pharmacologically diverse PDBbind
pockets, a state-of-the-art DrugFlow generator achieved \textbf{0\% HEW
site occupancy}---not a single generated molecule placed any atom within
2.5\,\AA{} of a HEW site---despite producing chemically valid, drug-like
molecules (QED $=0.39\pm0.14$) with reasonable Vina scores.  The
generator's implicit geometric prior fundamentally diverges from the
explicit physicochemical objective of water displacement.

Faced with this blindness, the most instinctively appealing strategy is
``hard-fix'': rigidly overwriting anchor-atom coordinates to enforce site
occupancy after each ODE integration
step~\cite{griesbacher2026ebmol,esfield2026}.  To evaluate this rigorously,
we turn to the \textbf{Kinetic Path Energy (KPE)} diagnostic
framework~\cite{li2026kpe}, which quantifies the transport cost of a
flow-matching trajectory.  Our KPE diagnosis reveals a devastating physical
picture: each coordinate overwrite ($\bm{x}_{\text{anchor}} \leftarrow
\bm{x}_0^{\text{anchor}}$) constitutes an instantaneous teleport in
configuration space,
$\Deltax_{\text{hard-fix}}^{(i)} = \bm{x}_0^{\text{anchor},(i)} - \xt^{(i)}$,
generating a velocity spike of effectively infinite magnitude.  As
quantified in Table~\ref{tab:kpe}, this injects \textbf{98.5\% excess
kinetic energy}---a 31,468-fold surge---directly \textbf{locks and tears
internal conformational degrees of freedom}, preventing the molecule from
relaxing into a globally optimal binding pose and producing the
\textbf{Occupancy--Affinity Paradox} (molecules with successfully placed
anchors exhibit systematically \emph{worse} Vina scores; 3mfw pocket:
$p=0.067$, Cliff's $\delta=+0.47$).  This is not an implementation flaw:
the root cause is a \textbf{fundamental dimensionality mismatch}---hard-fix
applies a $\reals^{3N_{\text{atoms}}}$-dimensional correction to achieve a
mere $\reals^3$ control objective (rigid translation), turning the
$3N_{\text{atoms}}-3$ excess degrees of freedom into channels for
injecting unbounded kinetic energy.  Even annealing strategies that
gradually release the constraint still inject 69.5\% excess energy
(Table~\ref{tab:kpe}), because irreversible KPE spikes from the early
hard-fix phase have already corrupted the trajectory.

To resolve this, we propose \textbf{Kinematic Anchor Guidance}, a principled
alternative that operates in the smallest subspace sufficient for the
control objective.  By orthogonally decomposing the guidance displacement
field and projecting the site-compatibility gradient exclusively onto the
centre-of-mass (CoM) subspace ($\reals^3$), our method mathematically
guarantees \textbf{zero strain} on all internal coordinates (bond lengths,
angles, torsions; Theorem~1, Section~\ref{sec:methods}).  The molecule is
softly pulled toward the target as a rigid body---\emph{attracted}, never
overwritten---preserving full internal flexibility for natural
conformational relaxation and inherently preventing steric clashes.  This
contrast highlights the fundamental trade-off in inference-time guidance:
hard-fix enforces exact coordinate matching at the cost of unbounded
kinetic energy injection ($\KPERatio = 98.5\%$), inevitably corrupting
the learned conformational prior, while kinematic anchoring achieves
the spatial targeting objective through bounded CoM-level attraction,
incurring negligible energy cost ($\KPERatio = 0.006\%$).

'''

compressed_intro += contrib_block

content = content[:intro_start] + compressed_intro + content[intro_end:]

# ═══════════════════════════════════════════════════════════════
# COMPRESSION 2: Related Work — trim each subsection to ~150 words
# ═══════════════════════════════════════════════════════════════

# Find Related Work boundaries
rw_start = content.find(r'\section{Related Work}')
methods_marker = r'\section{Methods}'
rw_end = content.find(methods_marker)

# Build compressed RW
compressed_rw = r'''\section{Related Work}

\subsection{3D Structure-Based Molecular Generation}

Structure-based \emph{de novo} molecular generation has been transformed by
deep generative models operating directly on 3D atomic coordinates,
including diffusion-based TargetDiff~\cite{guan2023targetdiff},
DiffSBDD~\cite{schneuing2024diffsbdd}, and
DecompDiff~\cite{guan2023decompdiff}, and flow-matching-based
DrugFlow~\cite{drugflow2024} and FLAG~\cite{zhang2024flag}.
\textbf{The root cause of a critical deficiency lies in the training data.}
PDB structures~\cite{berman2000pdb} rarely retain explicit interfacial
water molecules, so the learned distribution $p(\text{ligand} \mid
\text{pocket})$ inherently lacks the physicochemical prior for water
displacement---a data-induced \textbf{water-blindness} that leaves all
these methods without site-type-level control over water-occupied vs.\
water-displaceable regions~\cite{abel2008,michel2009}.

\textbf{Scope of applicability.}
Our method decomposes the continuous velocity field of flow-matching ODEs
and diffusion SDEs (Section~\ref{sec:methods}).  It is therefore
\textbf{agnostic to continuous-coordinate generative backbones} but
\emph{not} applicable to discrete-token autoregressive models, which lack
a continuous velocity-field structure.  Given the field's decisive shift
toward continuous-space diffusion and flow-matching paradigms, this
limitation does not constrain the method's broad applicability.

\subsection{Hydration Sites in Drug Design}

Water's thermodynamic role in protein--ligand binding is well
established~\cite{dunitz1994,ladbury1996}.  WaterMap~\cite{abel2008},
GIST~\cite{nguyen2012gist}, and GCMC~\cite{ross2012gcmc} quantify
hydration thermodynamics, while Michel et al.~\cite{michel2009} and Wang
et al.~\cite{wang2015water} characterised the energetic gains from
targeted water displacement.  Spyrakis et al.~\cite{spyrakis2017} called
for computational tools integrating water thermodynamics into molecular
design.  \textbf{Despite their accuracy, these tools produce static,
non-differentiable outputs}---heat maps and discrete $\Delta G$ values
decoupled from the generative process.  The \textbf{Kinetic Path Energy
(KPE) framework}~\cite{li2026kpe} was formalised to quantify flow-matching
transport costs; we repurpose it to isolate guidance-induced kinetic
energy, providing the quantitative foundation for our zero-strain
alternative (Section~\ref{sec:kpe_diagnosis}).

\subsection{Inference-Time Guidance for Molecular Generation}

Inference-time guidance injects design objectives into pre-trained
generators \textbf{without retraining}~\cite{dhariwal2021classifier,ho2022cfg,bansal2023universal}.
In the molecular domain, Lai et al.~\cite{lai2025force} incorporated
force-field gradients to reduce ligand strain, Lam et
al.~\cite{lam2026metadiffusion} introduced SVGD-based meta-energy biasing,
Li et al.~\cite{li2026kpe} formalised KPE and proposed velocity modulation
(KTS), and Griesbacher et al.~\cite{griesbacher2026ebmol} developed
atom-coordinate fixation via learned energy potentials.
\textbf{While effective as retraining-free interventions, these methods
share a fundamental limitation.}  They operate on global molecular
properties and, when local constraints are applied, lack a rigorous
kinematic decomposition to guarantee zero internal strain.  Although Li et
al.~\cite{li2026kpe} address velocity modulation globally and Lai et
al.~\cite{lai2025force} reduce strain through force-field terms, neither
provides a mathematical zero-strain guarantee for site-specific constraints.
Our kinematic anchor guidance addresses both gaps: site-type-level control
and mathematically guaranteed kinematic consistency, as quantified by
Table~\ref{tab:method_comparison}.

'''

content = content[:rw_start] + compressed_rw + content[rw_end:]

with open(TEX, 'w') as f:
    f.write(content)

# Report word counts
intro_words = len(compressed_intro.split())
rw_words = len(compressed_rw.split())
print(f'Compressed Intro: ~{intro_words} words')
print(f'Compressed RW: ~{rw_words} words')
print('Done. Compile with: cd /root/ESField/paper_latex && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex')
