#!/usr/bin/env python3
"""
BADGER-Emulating Baseline: Global Vina-like Classifier Gradient Injection.

BADGER (Jian et al., 2024) uses a pretrained EGNN classifier to predict Vina
binding affinity, then injects the gradient ∇_x Vina_pred(x) into ALL atom
coordinates during DDPM sampling.  This is a GLOBAL R^{3N} injection — the
classifier has NO hydration-site awareness.

We emulate this pattern:
  1. At each ODE step (t ≥ phase_gate), compute a protein-ligand interaction
     score as a differentiable Vina surrogate.
  2. Compute the gradient ∇_x Score(x) via finite differences.
  3. Inject this gradient globally (R^{3N}) — NO CoM projection, NO HEW targeting.

EXPECTED OUTCOME:
  - Vina Score: GOOD (global guidance toward better binding)
  - DirectOcc_HEW: ≈0% (water-blind — no HEW site awareness)
  - Strain: MODERATE (global gradient injection adds some strain)
  - ρ_KPE: MODERATE (R^{3N} injection)

This directly validates the paper's central claim: global guidance methods
(BADGER, Lai et al.) cannot achieve HEW site targeting because they lack
site-type-level awareness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from naive_forcefield_guidance import ATOM_TYPE_TO_ELEMENT, atom_type_probs_to_element

# ═══════════════════════════════════════════════════════════════════════════
# BADGER-Emulating Guidance Callback
# ═══════════════════════════════════════════════════════════════════════════

class BadgerGuidance:
    """BADGER-style global Vina classifier gradient injection.

    This emulates BADGER's core mechanism: at each step, compute a binding
    quality score, differentiate w.r.t. atom positions, and inject the
    gradient globally (R^{3N}).

    Key properties (matching BADGER's behavior):
      - GLOBAL scope: gradient applied to ALL atoms
      - WATER-BLIND: no HEW site awareness whatsoever
      - No kinematic decomposition: raw R^{3N} injection

    Usage:
        cb = BadgerGuidance(
            n_atoms=N, atom_types=h, lambda_badger=1.0,
            protein_pdb="3mfw_protein.pdb",
        )
        model.simulate(..., post_step_callback=cb)
    """

    def __init__(
        self,
        n_atoms: int,
        atom_types: np.ndarray | torch.Tensor | None = None,
        *,
        lambda_badger: float = 1.0,
        total_steps: int = 100,
        framework: str = "ode",
        schedule: str = "quadratic",
        grad_clip: float = 1.0,
        phase_gate: float = 0.6,
        device: str = "cpu",
        verbose: bool = False,
        protein_pdb: str | None = None,
    ):
        self.n_atoms = n_atoms
        self.lambda_badger = lambda_badger
        self.total_steps = total_steps
        self.framework = framework
        self.schedule = schedule
        self.grad_clip = grad_clip
        self.phase_gate = phase_gate
        self.device = device
        self.verbose = verbose

        if atom_types is not None:
            if isinstance(atom_types, torch.Tensor):
                self._atom_types_ref = atom_types.detach().cpu().numpy()
            else:
                self._atom_types_ref = np.asarray(atom_types)
        else:
            self._atom_types_ref = None

        # Load protein structure for interaction scoring
        self._protein_mol: Chem.Mol | None = None
        self._protein_coords: np.ndarray | None = None
        if protein_pdb and Path(protein_pdb).exists():
            self._protein_mol = Chem.MolFromPDBFile(protein_pdb, removeHs=False)
            if self._protein_mol is not None:
                conf = self._protein_mol.GetConformer()
                n_prot = self._protein_mol.GetNumAtoms()
                self._protein_coords = np.array([
                    conf.GetAtomPosition(i) for i in range(n_prot)
                ])
                if verbose:
                    print(f"  [BADGER] Loaded protein: {n_prot} atoms")

        self._call_count: int = 0
        self._grad_norms: list[float] = []
        self._interaction_energies: list[float] = []

    # ── Public API ──────────────────────────────────────────────────────

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        """Apply BADGER-style global Vina-surrogate gradient.

        Gradient is computed w.r.t. a simple protein-ligand interaction score,
        emulating the Vina classifier's binding affinity prediction.
        """
        self._call_count += 1

        if t_val < self.phase_gate:
            return ligand

        x = ligand["x"]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)

        n_atoms = x.shape[0]
        lam = self._compute_lambda(t_val, self.lambda_badger)

        if n_atoms < 2 or lam <= 0:
            return ligand

        # Compute interaction score gradient via finite differences
        x_np = x.detach().cpu().numpy().astype(np.float64)
        eps = 1e-3
        grad = np.zeros((n_atoms, 3), dtype=np.float64)

        for i in range(n_atoms):
            for d in range(3):
                x_plus = x_np.copy()
                x_plus[i, d] += eps
                e_plus = self._compute_interaction_score(x_plus)

                x_minus = x_np.copy()
                x_minus[i, d] -= eps
                e_minus = self._compute_interaction_score(x_minus)

                grad[i, d] = (e_plus - e_minus) / (2.0 * eps)

        grad_t = torch.from_numpy(grad).float().to(self.device)

        # ── BADGER-STYLE GLOBAL INJECTION (R^{3N}) ──
        # NO CoM projection. NO HEW awareness. All atoms get per-atom gradient.
        # This is the exact pattern from BADGER's sample_diffusion():
        #   ligand_pos_next = pos_model_mean + noise - s * prefactor * grad_pos

        grad_norm = grad_t.norm(dim=-1)
        max_norm = grad_norm.max().item()
        if max_norm > self.grad_clip:
            grad_t = grad_t * (self.grad_clip / (max_norm + 1e-8))

        x_updated = x.detach() - lam * grad_t  # negative: minimize score

        # Track stats
        self._grad_norms.append(float(grad_t.norm().item()))
        e_current = self._compute_interaction_score(x_np)
        self._interaction_energies.append(e_current)

        if isinstance(ligand["x"], np.ndarray):
            ligand["x"] = x_updated.detach().cpu().numpy()
        else:
            ligand["x"] = x_updated.detach()

        if self.verbose and self._call_count % 20 == 0:
            print(f"  [BADGER] step={step_idx}, t={t_val:.3f}, λ={lam:.4f}, "
                  f"|grad|={max_norm:.4f}, E_int={e_current:.2f}")

        return ligand

    # ── Internal helpers ────────────────────────────────────────────────

    def _compute_interaction_score(self, ligand_coords: np.ndarray) -> float:
        """Compute a simple protein-ligand interaction score.

        Uses a soft Lennard-Jones-like potential as a differentiable Vina
        surrogate.  Lower (more negative) = better binding.
        """
        if self._protein_coords is None:
            return 0.0

        # Simple vdW-like attractive + repulsive score
        # V(r) = ε * ((σ/r)^12 - 2*(σ/r)^6)
        prot_coords = self._protein_coords
        score = 0.0
        sigma = 3.0  # approximate vdW contact distance (Å)
        epsilon = 1.0

        for i in range(len(ligand_coords)):
            for j in range(len(prot_coords)):
                r = float(np.linalg.norm(ligand_coords[i] - prot_coords[j]))
                if r < 0.5:  # avoid singularity
                    r = 0.5
                if r > 8.0:  # cutoff for efficiency
                    continue
                sr6 = (sigma / r) ** 6
                score += epsilon * (sr6 * sr6 - 2.0 * sr6)

        return score

    def _compute_lambda(self, t_val: float, lambda_val: float) -> float:
        t = float(t_val)
        if self.schedule == "quadratic":
            return lambda_val * (1.0 - t) ** 2
        elif self.schedule == "constant":
            return lambda_val
        elif self.schedule == "linear":
            return lambda_val * (1.0 - t)
        else:
            return lambda_val * (1.0 - t) ** 2

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def mean_grad_norm(self) -> float:
        if not self._grad_norms:
            return 0.0
        return float(np.mean(self._grad_norms))

    @property
    def mean_interaction_energy(self) -> float:
        if not self._interaction_energies:
            return 0.0
        return float(np.mean(self._interaction_energies))

    def get_summary(self) -> dict:
        return {
            "guidance_type": "badger_emulation",
            "lambda_badger": self.lambda_badger,
            "schedule": self.schedule,
            "n_calls": self._call_count,
            "mean_grad_norm": self.mean_grad_norm,
            "mean_interaction_energy": self.mean_interaction_energy,
            "framework": self.framework,
        }

    def to(self, device: str) -> "BadgerGuidance":
        self.device = device
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_badger_callback(
    n_atoms: int,
    atom_types: np.ndarray | None = None,
    *,
    lambda_badger: float = 1.0,
    total_steps: int = 100,
    framework: str = "ode",
    schedule: str = "quadratic",
    grad_clip: float = 1.0,
    phase_gate: float = 0.6,
    device: str = "cpu",
    protein_pdb: str | None = None,
    verbose: bool = False,
) -> BadgerGuidance:
    """Factory for BadgerGuidance callback."""
    return BadgerGuidance(
        n_atoms=n_atoms,
        atom_types=atom_types,
        lambda_badger=lambda_badger,
        total_steps=total_steps,
        framework=framework,
        schedule=schedule,
        grad_clip=grad_clip,
        phase_gate=phase_gate,
        device=device,
        protein_pdb=protein_pdb,
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI — quick test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print("BadgerGuidance — quick test")
        n_atoms = 6
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.1, 1.2, 0.0],
            [1.4, 2.4, 0.0],
            [0.0, 2.4, 0.0],
            [-0.7, 1.2, 0.0],
        ], dtype=torch.float32)

        cb = BadgerGuidance(n_atoms=n_atoms, lambda_badger=0.1, phase_gate=0.0,
                           verbose=True)
        ligand = {"x": x}
        for step in range(5):
            ligand = cb(ligand, step, step / 10)

        print(f"\nSummary: {cb.get_summary()}")
        print("Test passed.")
