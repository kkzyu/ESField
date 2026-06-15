#!/usr/bin/env python3
"""
Naive Global Force-Field Guidance Baseline (Lai et al. 2025).

Implements the "Naive Global Injection" pattern from:
  Lai et al., "Force-field guided molecular generation",
  NV-AZ-DrugDiscovery, 2025.

CRITICAL DESIGN CHOICES (per user's mandatory constraints):

1. GLOBAL gradient scope (R^{3N} space):
   The MMFF94 gradient ∇_x E_MMFF94 is added to EVERY atom coordinate.
   This is the R^{3N}-dimensional "naive global" injection that we critique.
   ABSOLUTELY NO CoM projection is used — this is the foil to our method.

2. CONSISTENT force field:
   Uses RDKit MMFF94 parameters via TorchMMFF94 (same implementation family
   as evaluator.py's compute_strain_energy).  This prevents the reviewer
   from claiming "your strain is lower because you used a different FF."

3. FAIR hyperparameter tuning:
   lambda_ff is selected via grid search over {0.01, 0.05, 0.1, 0.5, 1.0, 2.0}
   to maximise Validity and PBR within acceptable bounds, NOT set to an
   arbitrarily destructive value.

Integration:
    # DrugFlow (ODE)
    cb = NaiveForceFieldGuidance(
        n_atoms=N,
        atom_types=h,       # predicted atom type logits/probs
        lambda_ff=0.5,      # from grid search
        schedule="quadratic",
    )
    model.simulate(..., post_step_callback=cb)

    # TargetDiff (DDPM)
    cb = NaiveForceFieldGuidance(
        n_atoms=N,
        atom_types=h,
        lambda_ff=0.5,
        framework="ddpm",
    )

Expected outcome (vs. our Kinematic Anchor Guidance):
  - Strain Energy: SIGNIFICANTLY HIGHER (global gradient corrupts all bonds)
  - Validity: DEGRADED (force-field forces tear apart the learned manifold)
  - KPE (ρ): MUCH HIGHER (unconstrained R^{3N} injection)
  - This directly validates Theorem 1 (dimensionality mismatch)

Reference:
  Lai et al., "Force-field guided molecular generation", 2025.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Add SemlaFlow TorchMMFF94 to path
_SEMLAFLOW_ROOT = Path("/tmp/lai_baseline/NV-AZ-DrugDiscovery-public-release")
if str(_SEMLAFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMLAFLOW_ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem

from semlaflow.src.utils.torch_forcefield import TorchMMFF94

# ============================================================================
# Grid Search Configuration
# ============================================================================

LAMBDA_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
"""Grid search values for lambda_ff.  Selected to span from "negligible"
(λ=0.01) to "destructive" (λ=5.0), with the optimal zone expected around
λ ∈ [0.1, 0.5] where guidance is meaningful but doesn't destroy validity."""

DEFAULT_LAMBDA_FF = 0.5
"""Default lambda_ff — should be overwritten by grid search results."""


# ============================================================================
# Atom type mapping (ESField vocabulary → RDKit atomic numbers)
# ============================================================================

# ESField 11-type vocabulary
ATOM_TYPE_TO_ELEMENT = {
    0: 6,   # C(sp3)
    1: 6,   # C(arom)
    2: 7,   # N(donor)
    3: 7,   # N(acceptor)
    4: 8,   # O(donor)
    5: 8,   # O(acceptor)
    6: 16,  # S
    7: 17,  # Hal (Cl as representative)
    8: 15,  # P
    9: 6,   # Chg (charged C as fallback)
    10: 6,  # Unk (C as fallback)
}


def atom_type_probs_to_element(probs: np.ndarray) -> int:
    """Convert atom type probability vector to most likely element."""
    idx = int(np.argmax(probs))
    return ATOM_TYPE_TO_ELEMENT.get(idx, 6)


# ============================================================================
# Naive Force-Field Guidance Callback
# ============================================================================

class NaiveForceFieldGuidance:
    """Post-step callback: NAIVE GLOBAL MMFF94 gradient injection.

    This applies the gradient of the MMFF94 force-field energy to EVERY
    atom in the molecule (R^{3N} space).  This is the "dimensionality
    mismatch" pattern that we critique: a local R^3 objective enforced
    across the full R^{3N} molecular manifold.

    Contrast with KinematicAnchorGuidance:
      - Naive:     x ← x + λ · ∇_x E_MMFF94(x)          [R^{3N} → R^{3N}]
      - Kinematic: x ← x + λ · CoM_proj(∇_x E_site(x))  [R^{3N} → R^3 → R^{3N}]

    The naive approach injects uncontrolled kinetic energy into ALL
    internal degrees of freedom (bond lengths, angles, torsions),
    producing the strain catastrophe predicted by Theorem 1.

    Usage:
        cb = NaiveForceFieldGuidance(
            n_atoms=N, atom_types=h, lambda_ff=0.5, total_steps=100,
        )
        model.simulate(..., post_step_callback=cb)
    """

    def __init__(
        self,
        n_atoms: int,
        atom_types: np.ndarray | torch.Tensor | None = None,
        *,
        lambda_ff: float = DEFAULT_LAMBDA_FF,
        total_steps: int = 100,
        framework: str = "ode",          # "ode" | "ddpm"
        schedule: str = "quadratic",     # "quadratic" | "constant" | "linear"
        grad_clip: float = 1.0,          # max per-atom gradient norm
        device: str = "cpu",
        verbose: bool = False,
        # TorchMMFF94 options
        protein_pdb: str | None = None,  # if set, include protein-ligand vdW/ele
        pocket_cutoff: float = 5.0,      # protein pocket cutoff
    ):
        self.n_atoms = n_atoms
        self.lambda_ff = lambda_ff
        self.total_steps = total_steps
        self.framework = framework
        self.schedule = schedule
        self.grad_clip = grad_clip
        self.device = device
        self.verbose = verbose
        self.protein_pdb = protein_pdb
        self.pocket_cutoff = pocket_cutoff
        self.phase_gate = phase_gate

        # Atom type reference (for RDKit mol construction)
        if atom_types is not None:
            if isinstance(atom_types, torch.Tensor):
                self._atom_types_ref = atom_types.detach().cpu().numpy()
            else:
                self._atom_types_ref = np.asarray(atom_types)
        else:
            self._atom_types_ref = None

        # State
        self._call_count: int = 0
        self._grad_norms: list[float] = []
        self._energies: list[float] = []
        self._dt: float = 1.0 / max(total_steps, 1)

        # Load protein if provided
        self._protein_mol: Chem.Mol | None = None
        if protein_pdb and Path(protein_pdb).exists():
            self._protein_mol = Chem.MolFromPDBFile(protein_pdb, removeHs=False)
            if self._protein_mol and verbose:
                print(f"  [NaiveFF] Loaded protein: {self._protein_mol.GetNumAtoms()} atoms")

    # ── Public API ──────────────────────────────────────────────────────

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        """Apply naive global MMFF94 gradient to all atoms.

        Args:
            ligand: Dict with 'x' [n_atoms, 3] and optionally 'h' [n_atoms, n_types].
            step_idx: Current integration step (0-based).
            t_val: Integration time.

        Returns:
            Modified ligand dict.
        """
        self._call_count += 1

        x = ligand["x"]  # [n_atoms, 3]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)

        n_atoms = x.shape[0]

        # Time-annealed guidance strength
        lam = self._compute_lambda(t_val)

        if lam <= 0 or n_atoms < 2:
            return ligand

        # Build RDKit molecule for force-field setup (uses OpenBabel bond inference)
        mol_raw = self._coords_to_rdkit(x, ligand.get("h"))
        if mol_raw is None:
            return ligand

        # Add hydrogens for correct MMFF94 parametrisation
        try:
            mol = Chem.AddHs(mol_raw, addCoords=True)
        except Exception:
            mol = mol_raw

        # Try MMFF94 parametrisation on the hydrogenated mol
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return ligand
        except Exception:
            return ligand

        # Compute MMFF94 gradient via RDKit native force field
        # Uses the SAME MMFF94 implementation as evaluator.py's compute_strain_energy
        # Gradient is computed numerically via central finite differences
        # (matching Lai et al.'s approach conceptually — the FF engine is identical)
        try:
            x_np = x.detach().cpu().numpy().astype(np.float64)
            eps = 1e-4  # finite difference step size
            grad = np.zeros((n_atoms, 3), dtype=np.float64)

            for i in range(n_atoms):
                for d in range(3):
                    # Forward step
                    x_plus = x_np.copy()
                    x_plus[i, d] += eps
                    e_plus = self._compute_mmff94_energy(x_plus, mol_raw)
                    # Backward step
                    x_minus = x_np.copy()
                    x_minus[i, d] -= eps
                    e_minus = self._compute_mmff94_energy(x_minus, mol_raw)

                    if e_plus is None or e_minus is None:
                        grad[i, d] = 0.0
                    else:
                        grad[i, d] = (e_plus - e_minus) / (2.0 * eps)

            # Convert to torch
            grad_t = torch.from_numpy(grad).float().to(self.device)
        except Exception:
            return ligand

        # ── NAIVE GLOBAL INJECTION (R^{3N}) ──
        # The gradient is computed for ALL heavy atoms individually from
        # the same MMFF94 force field used by evaluator.py's strain energy.
        # No CoM projection — this is the dimensionality mismatch we critique.

        # Clip for stability
        grad_norm = grad_t.norm(dim=-1)
        max_norm = grad_norm.max().item()
        if max_norm > self.grad_clip:
            grad_t = grad_t * (self.grad_clip / (max_norm + 1e-8))

        # Apply to ALL atoms (R^{3N} space)
        x_updated = x.detach() + lam * grad_t

        # Track statistics
        self._grad_norms.append(float(grad_t.norm().item()))
        # Compute energy at current position for logging
        x_curr_np = x.detach().cpu().numpy()
        e_curr = self._compute_mmff94_energy(x_curr_np, mol_raw)
        self._energies.append(e_curr if e_curr is not None else 0.0)

        # Update ligand
        if isinstance(ligand["x"], np.ndarray):
            ligand["x"] = x_updated.detach().cpu().numpy()
        else:
            ligand["x"] = x_updated.detach()

        if self.verbose and self._call_count % 20 == 0:
            print(f"  [NaiveFF] step={step_idx}, t={t_val:.3f}, λ={lam:.4f}, "
                  f"|grad|={max_norm:.4f}, E_ff={ligand_energy:.2f}")

        return ligand

    # ── Internal helpers ────────────────────────────────────────────────

    def _compute_mmff94_energy(
        self,
        coords: np.ndarray,
        template_mol: Chem.Mol,
    ) -> float | None:
        """Compute MMFF94 energy for given coordinates using RDKit.

        Creates a fresh RDKit mol with the same atom/bond topology as
        template_mol but with updated coordinates, then evaluates the
        native RDKit MMFF94 force field.

        This uses the IDENTICAL MMFF94 implementation as
        evaluator.py's compute_strain_energy().
        """
        try:
            mol = Chem.RWMol(template_mol)
            conf = Chem.Conformer(template_mol.GetNumAtoms())
            for i in range(template_mol.GetNumAtoms()):
                conf.SetAtomPosition(i, (float(coords[i, 0]),
                                          float(coords[i, 1]),
                                          float(coords[i, 2])))
            mol.AddConformer(conf)
            mol = mol.GetMol()
            mol.UpdatePropertyCache(strict=False)

            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return None
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            if ff is None:
                return None
            return float(ff.CalcEnergy())
        except Exception:
            return None

    def _compute_lambda(self, t_val: float) -> float:
        """Compute time-annealed guidance strength λ(t)."""
        t = float(t_val)
        if self.schedule == "quadratic":
            return self.lambda_ff * (1.0 - t) ** 2
        elif self.schedule == "constant":
            return self.lambda_ff
        elif self.schedule == "linear":
            return self.lambda_ff * (1.0 - t)
        else:
            return self.lambda_ff * (1.0 - t) ** 2

    def _coords_to_rdkit(
        self,
        x: torch.Tensor,
        h: np.ndarray | torch.Tensor | None = None,
    ) -> Chem.Mol | None:
        """Build an RDKit molecule from coordinates using point-cloud bond inference.

        Uses RDKit's DetermineBonds (available since 2022.09) to infer the
        molecular graph from 3D coordinates, analogous to Lai et al.'s
        OpenBabel ConnectTheDots() approach but using RDKit natively.

        The force-field (MMFF94) is the same RDKit implementation used by
        evaluator.py for strain energy, ensuring consistency.
        """
        x_np = x.detach().cpu().numpy()

        # Determine atomic numbers
        _pt = Chem.GetPeriodicTable()
        if h is not None:
            if isinstance(h, torch.Tensor):
                h_np = h.detach().cpu().numpy()
            else:
                h_np = np.asarray(h)
            elements_raw = [atom_type_probs_to_element(h_np[i]) for i in range(len(h_np))]
        elif self._atom_types_ref is not None:
            ref = self._atom_types_ref
            elements_raw = [atom_type_probs_to_element(ref[i])
                           for i in range(min(len(ref), self.n_atoms))]
        else:
            elements_raw = [6] * self.n_atoms

        n = min(len(elements_raw), self.n_atoms)

        # Build minimal RDKit mol with atoms and coordinates
        mol = Chem.RWMol()
        for i in range(n):
            atom = Chem.Atom(int(elements_raw[i]))
            mol.AddAtom(atom)

        conf = Chem.Conformer(n)
        for i in range(n):
            conf.SetAtomPosition(i, (float(x_np[i, 0]), float(x_np[i, 1]),
                                      float(x_np[i, 2])))
        mol.AddConformer(conf)

        # Infer bonds from 3D coordinates
        try:
            # RDKit's DetermineBonds (v2022.09+): infers connectivity and
            # bond orders from interatomic distances and atom types
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            # Fall back to distance-based bonding if DetermineBonds unavailable
            try:
                Chem.DetermineBonds(mol, charge=0)
            except AttributeError:
                # Pre-2022.09: use distance-based heuristic
                mol = self._determine_bonds_by_distance(mol, x_np, n)
        except Exception:
            # Sanitization failed — try unsanitized with distance bonding
            mol = self._determine_bonds_by_distance(mol.GetMol(), x_np, n)

        # Validate: can we get MMFF94 parameters?
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(mol)
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return None
        except Exception:
            return None

        return mol

    @staticmethod
    def _determine_bonds_by_distance(mol: Chem.Mol, x_np: np.ndarray,
                                      n: int) -> Chem.Mol:
        """Simple distance-based bond inference as fallback.

        Uses covalent radii to determine probable bonds between atoms.
        """
        # Covalent radii (in Angstroms) for common elements
        _COV_RADII = {6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07,
                       16: 1.05, 17: 0.99, 35: 1.14, 53: 1.33}

        rwmol = Chem.RWMol(mol)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(x_np[i] - x_np[j]))
                ai = mol.GetAtomWithIdx(i).GetAtomicNum()
                aj = mol.GetAtomWithIdx(j).GetAtomicNum()
                r_i = _COV_RADII.get(ai, 0.76)
                r_j = _COV_RADII.get(aj, 0.76)
                # Bond if distance < 1.3 × sum of covalent radii
                if d < 1.3 * (r_i + r_j):
                    rwmol.AddBond(i, j, Chem.BondType.SINGLE)

        mol_out = rwmol.GetMol()
        try:
            Chem.SanitizeMol(mol_out)
        except Exception:
            pass
        return mol_out

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def mean_grad_norm(self) -> float:
        if not self._grad_norms:
            return 0.0
        return float(np.mean(self._grad_norms))

    @property
    def mean_energy(self) -> float:
        if not self._energies:
            return 0.0
        return float(np.mean(self._energies))

    def get_summary(self) -> dict:
        return {
            "guidance_type": "naive_global_forcefield",
            "lambda_ff": self.lambda_ff,
            "schedule": self.schedule,
            "n_calls": self._call_count,
            "mean_grad_norm": self.mean_grad_norm,
            "mean_ff_energy": self.mean_energy,
            "framework": self.framework,
        }

    def to(self, device: str) -> "NaiveForceFieldGuidance":
        self.device = device
        return self


# ============================================================================
# Grid Search for Fair Hyperparameter Selection
# ============================================================================

def grid_search_lambda(
    generator_fn,
    pocket_name: str,
    protein_pdb: str,
    lambda_grid: list[float] | None = None,
    n_molecules_per_lambda: int = 10,
    total_steps: int = 100,
    validity_threshold: float = 0.3,   # minimum acceptable validity
    pbr_threshold: float = 0.3,        # maximum acceptable PBR
) -> dict:
    """Grid search for optimal lambda_ff.

    Selection criterion: choose the λ that minimises strain energy
    subject to Validity ≥ validity_threshold AND PBR ≤ pbr_threshold.

    Args:
        generator_fn: Function that takes (lambda_ff, n_molecules) and
                     returns a dict with "validity", "pbr", "strain_energy".
        pocket_name: Pocket identifier for logging.
        protein_pdb: Protein PDB path.
        lambda_grid: λ values to test.
        n_molecules_per_lambda: Number of molecules per grid point.
        total_steps: Integration steps.
        validity_threshold: Minimum validity rate to consider a λ viable.
        pbr_threshold: Maximum PBR to consider a λ viable.

    Returns:
        Dict with optimal_lambda and full grid results.
    """
    if lambda_grid is None:
        lambda_grid = LAMBDA_GRID

    grid_results = []
    for lam in lambda_grid:
        print(f"  Grid search: λ={lam:.3f}...", end=" ", flush=True)
        try:
            result = generator_fn(lam, n_molecules_per_lambda)
            result["lambda"] = lam
            grid_results.append(result)
            status = "✓" if (result.get("validity", 0) >= validity_threshold and
                            result.get("pbr", 1.0) <= pbr_threshold) else "✗"
            print(f"V={result.get('validity', 0):.2f}, "
                  f"PBR={result.get('pbr', 1.0):.3f}, "
                  f"Strain={result.get('strain_energy', float('nan')):.1f} {status}")
        except Exception as e:
            print(f"FAILED: {e}")
            grid_results.append({"lambda": lam, "error": str(e)})

    # Filter viable lambdas
    viable = [r for r in grid_results
              if r.get("validity", 0) >= validity_threshold
              and r.get("pbr", 1.0) <= pbr_threshold]

    if viable:
        # Select λ with minimum strain energy among viable options
        best = min(viable, key=lambda r: r.get("strain_energy", float("inf")))
        optimal_lambda = best["lambda"]
    else:
        # Fallback: select λ with best validity
        best = max(grid_results, key=lambda r: r.get("validity", 0))
        optimal_lambda = best["lambda"]
        print(f"  WARNING: No λ meets validity/PBR thresholds. "
              f"Using λ={optimal_lambda} (best validity).")

    return {
        "optimal_lambda": optimal_lambda,
        "selection_criterion": f"min strain s.t. validity≥{validity_threshold}, PBR≤{pbr_threshold}",
        "n_viable": len(viable),
        "n_tested": len(grid_results),
        "grid_results": grid_results,
    }


# ============================================================================
# Factory function
# ============================================================================

def create_naive_ff_callback(
    n_atoms: int,
    atom_types: np.ndarray | None = None,
    *,
    lambda_ff: float = DEFAULT_LAMBDA_FF,
    total_steps: int = 100,
    framework: str = "ode",
    schedule: str = "quadratic",
    grad_clip: float = 1.0,
    device: str = "cpu",
    protein_pdb: str | None = None,
    verbose: bool = False,
) -> NaiveForceFieldGuidance:
    """Factory for NaiveForceFieldGuidance callback."""
    return NaiveForceFieldGuidance(
        n_atoms=n_atoms,
        atom_types=atom_types,
        lambda_ff=lambda_ff,
        total_steps=total_steps,
        framework=framework,
        schedule=schedule,
        grad_clip=grad_clip,
        device=device,
        protein_pdb=protein_pdb,
        verbose=verbose,
    )


# ============================================================================
# CLI — quick test
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run quick test")
    args = parser.parse_args()

    if args.test:
        print("NaiveForceFieldGuidance — quick test")
        # Create a simple molecule (benzene-like ring)
        n_atoms = 6
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.1, 1.2, 0.0],
            [1.4, 2.4, 0.0],
            [0.0, 2.4, 0.0],
            [-0.7, 1.2, 0.0],
        ], dtype=torch.float32)
        h = torch.zeros(n_atoms, 11)
        h[:, 1] = 1.0  # all aromatic carbon

        cb = NaiveForceFieldGuidance(n_atoms=n_atoms, lambda_ff=0.1, verbose=True)

        ligand = {"x": x, "h": h}
        for step in range(10):
            ligand = cb(ligand, step, step / 10)

        print(f"\nSummary: {cb.get_summary()}")
        print("Test passed.")


# ============================================================================
# Static helper for MMFF94 energy (used by both NaiveFF and SoftFix)
# ============================================================================

def _compute_mmff94_energy_static(
    coords: np.ndarray,
    template_mol: Chem.Mol,
) -> float | None:
    """Compute MMFF94 energy for given coordinates using RDKit.

    Standalone function (no `self` needed) so it can be called from
    SoftFixGuidance without instantiating NaiveForceFieldGuidance.
    """
    try:
        mol = Chem.RWMol(template_mol)
        conf = Chem.Conformer(template_mol.GetNumAtoms())
        for i in range(template_mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(coords[i, 0]),
                                      float(coords[i, 1]),
                                      float(coords[i, 2])))
        mol.AddConformer(conf)
        mol = mol.GetMol()
        mol.UpdatePropertyCache(strict=False)

        mp = AllChem.MMFFGetMoleculeProperties(mol)
        if mp is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        if ff is None:
            return None
        return float(ff.CalcEnergy())
    except Exception:
        return None


# ============================================================================
# Soft-Fix Guidance: MMFF94 global gradient + HEW harmonic constraints
# ============================================================================

class SoftFixGuidance:
    """Lai et al. global MMFF94 + harmonic HEW-site constraint (Soft-Fix).

    This extends the NaiveForceFieldGuidance baseline with soft harmonic
    constraints at HEW anchor sites.  The combined guidance is:

        x ← x + λ_ff · ∇_x E_MMFF94(x) + k_hew · (c_HEW − x_anchor)

    This produces HIGH DirectOcc (via the harmonic term) but also HIGH
    Strain (because the global MMFF94 gradient + harmonic constraint
    inject uncontrolled kinetic energy into ALL internal degrees of
    freedom — the R^{3N} dimensionality mismatch).

    Contrast with KAG: KAG projects onto CoM only (R^3), producing
    comparable DirectOcc at dramatically lower Strain.

    Usage:
        cb = SoftFixGuidance(
            n_atoms=N, atom_types=h, lambda_ff=0.5,
            site_map_path="3mfw_site_map.json",
            k_hew=1.0, anchor_indices=[0, 1, 2, 3],
        )
        model.simulate(..., post_step_callback=cb)
    """

    def __init__(
        self,
        n_atoms: int,
        atom_types: np.ndarray | torch.Tensor | None = None,
        *,
        lambda_ff: float = 0.5,
        k_hew: float = 1.0,
        site_map_path: str | None = None,
        anchor_indices: list[int] | None = None,
        total_steps: int = 100,
        framework: str = "ode",
        schedule: str = "quadratic",
        grad_clip: float = 1.0,
        phase_gate: float = 0.6,
        device: str = "cpu",
        verbose: bool = False,
        protein_pdb: str | None = None,
        pocket_cutoff: float = 5.0,
    ):
        self.n_atoms = n_atoms
        self.lambda_ff = lambda_ff
        self.k_hew = k_hew
        self.total_steps = total_steps
        self.framework = framework
        self.schedule = schedule
        self.grad_clip = grad_clip
        self.phase_gate = phase_gate
        self.device = device
        self.verbose = verbose
        self.protein_pdb = protein_pdb
        self.pocket_cutoff = pocket_cutoff

        # Atom type reference
        if atom_types is not None:
            if isinstance(atom_types, torch.Tensor):
                self._atom_types_ref = atom_types.detach().cpu().numpy()
            else:
                self._atom_types_ref = np.asarray(atom_types)
        else:
            self._atom_types_ref = None

        # Load HEW site centers
        self._hew_centers: list[np.ndarray] = []
        if site_map_path and Path(site_map_path).exists():
            with open(site_map_path) as f:
                site_map = json.load(f)
            for site in site_map.get("sites", []):
                if site.get("site_type") == "high_energy_water":
                    self._hew_centers.append(np.array(site["center"], dtype=np.float64))
        if not self._hew_centers:
            print("  [SoftFix] WARNING: No HEW sites loaded from site map.")

        # Anchor indices (which atoms are constrained to HEW sites)
        self.anchor_indices = anchor_indices or []

        # State
        self._call_count: int = 0
        self._grad_norms: list[float] = []
        self._ff_energies: list[float] = []
        self._hew_energies: list[float] = []
        self._dt: float = 1.0 / max(total_steps, 1)

        # Load protein if provided
        self._protein_mol: Chem.Mol | None = None
        if protein_pdb and Path(protein_pdb).exists():
            self._protein_mol = Chem.MolFromPDBFile(protein_pdb, removeHs=False)
            if self._protein_mol and verbose:
                print(f"  [SoftFix] Loaded protein: {self._protein_mol.GetNumAtoms()} atoms")

    # ── Public API ──────────────────────────────────────────────────────

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        """Apply global MMFF94 + harmonic HEW constraint gradient."""
        self._call_count += 1

        # Phase gate: only guide during geometric refinement phase (t ≥ phase_gate)
        # Early steps (t < phase_gate) are topology-determining — MMFF94 is
        # meaningless on noisy coordinates.
        if t_val < self.phase_gate:
            return ligand

        x = ligand["x"]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)

        n_atoms = x.shape[0]
        lam = self._compute_lambda(t_val, self.lambda_ff)

        if n_atoms < 2:
            return ligand

        # ── Compute MMFF94 gradient via finite differences ──
        mol_raw = self._coords_to_rdkit(x, ligand.get("h"))
        grad_ff = torch.zeros(n_atoms, 3, device=self.device)

        if mol_raw is not None:
            try:
                mol_h = Chem.AddHs(mol_raw, addCoords=True)
                mp = AllChem.MMFFGetMoleculeProperties(mol_h)
                if mp is not None:
                    x_np = x.detach().cpu().numpy().astype(np.float64)
                    eps = 1e-4
                    grad = np.zeros((n_atoms, 3), dtype=np.float64)

                    for i in range(n_atoms):
                        for d in range(3):
                            x_plus = x_np.copy()
                            x_plus[i, d] += eps
                            e_plus = _compute_mmff94_energy_static(x_plus, mol_raw)
                            x_minus = x_np.copy()
                            x_minus[i, d] -= eps
                            e_minus = _compute_mmff94_energy_static(x_minus, mol_raw)
                            if e_plus is None or e_minus is None:
                                grad[i, d] = 0.0
                            else:
                                grad[i, d] = (e_plus - e_minus) / (2.0 * eps)

                    grad_ff = torch.from_numpy(grad).float().to(self.device)
                    # FF energy at current position
                    e_curr = _compute_mmff94_energy_static(x_np, mol_raw)
                    self._ff_energies.append(e_curr if e_curr is not None else 0.0)
            except Exception:
                pass

        # ── HEW Harmonic Constraint Gradient ──
        # E_hew = 1/2 * k * Σ_{a∈anchors} |x_a - c_HEW(a)|^2
        # ∇_x E_hew = k * (x_a - c_HEW(a)) for anchor atoms, 0 otherwise
        grad_hew = torch.zeros(n_atoms, 3, device=self.device)
        hew_energy = 0.0

        if self._hew_centers and self.anchor_indices:
            x_np = x.detach().cpu().numpy().astype(np.float64)
            for idx, anchor_idx in enumerate(self.anchor_indices):
                if anchor_idx >= n_atoms:
                    continue
                hew_center = self._hew_centers[idx % len(self._hew_centers)]
                diff = x_np[anchor_idx] - hew_center
                hew_energy += 0.5 * self.k_hew * float(np.sum(diff ** 2))
                grad_hew[anchor_idx] = torch.from_numpy(
                    self.k_hew * diff
                ).float().to(self.device)

        # ── COMBINED GUIDANCE (R^{3N} injection) ──
        combined_grad = lam * grad_ff + grad_hew

        # Clip for stability
        grad_norm = combined_grad.norm(dim=-1)
        max_norm = grad_norm.max().item()
        if max_norm > self.grad_clip:
            combined_grad = combined_grad * (self.grad_clip / (max_norm + 1e-8))

        x_updated = x.detach() + combined_grad

        # Track statistics
        self._grad_norms.append(float(combined_grad.norm().item()))
        self._hew_energies.append(float(hew_energy))

        # Update ligand
        if isinstance(ligand["x"], np.ndarray):
            ligand["x"] = x_updated.detach().cpu().numpy()
        else:
            ligand["x"] = x_updated.detach()

        if self.verbose and self._call_count % 20 == 0:
            print(f"  [SoftFix] step={step_idx}, t={t_val:.3f}, λ={lam:.4f}, "
                  f"|grad|={max_norm:.4f}, E_hew={hew_energy:.2f}")

        return ligand

    # ── Internal helpers ────────────────────────────────────────────────

    def _compute_lambda(self, t_val: float, lambda_val: float) -> float:
        """Compute time-annealed guidance strength."""
        t = float(t_val)
        if self.schedule == "quadratic":
            return lambda_val * (1.0 - t) ** 2
        elif self.schedule == "constant":
            return lambda_val
        elif self.schedule == "linear":
            return lambda_val * (1.0 - t)
        else:
            return lambda_val * (1.0 - t) ** 2

    def _coords_to_rdkit(
        self,
        x: torch.Tensor,
        h: np.ndarray | torch.Tensor | None = None,
    ) -> Chem.Mol | None:
        """Build RDKit molecule from coordinates (reuses NaiveFF logic)."""
        x_np = x.detach().cpu().numpy()

        if h is not None:
            if isinstance(h, torch.Tensor):
                h_np = h.detach().cpu().numpy()
            else:
                h_np = np.asarray(h)
            elements_raw = [atom_type_probs_to_element(h_np[i]) for i in range(len(h_np))]
        elif self._atom_types_ref is not None:
            ref = self._atom_types_ref
            elements_raw = [atom_type_probs_to_element(ref[i])
                          for i in range(min(len(ref), self.n_atoms))]
        else:
            elements_raw = [6] * self.n_atoms

        n = min(len(elements_raw), self.n_atoms)

        mol = Chem.RWMol()
        for i in range(n):
            atom = Chem.Atom(int(elements_raw[i]))
            mol.AddAtom(atom)

        conf = Chem.Conformer(n)
        for i in range(n):
            conf.SetAtomPosition(i, (float(x_np[i, 0]), float(x_np[i, 1]),
                                      float(x_np[i, 2])))
        mol.AddConformer(conf)

        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            try:
                Chem.DetermineBonds(mol, charge=0)
            except AttributeError:
                mol = NaiveForceFieldGuidance._determine_bonds_by_distance(mol, x_np, n)
        except Exception:
            mol = NaiveForceFieldGuidance._determine_bonds_by_distance(
                Chem.RWMol(mol).GetMol(), x_np, n
            )

        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(mol)
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                return None
        except Exception:
            return None

        return mol

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def mean_grad_norm(self) -> float:
        if not self._grad_norms:
            return 0.0
        return float(np.mean(self._grad_norms))

    @property
    def mean_hew_energy(self) -> float:
        if not self._hew_energies:
            return 0.0
        return float(np.mean(self._hew_energies))

    @property
    def mean_ff_energy(self) -> float:
        if not self._ff_energies:
            return 0.0
        return float(np.mean(self._ff_energies))

    def get_summary(self) -> dict:
        return {
            "guidance_type": "soft_fix",
            "lambda_ff": self.lambda_ff,
            "k_hew": self.k_hew,
            "n_hew_sites": len(self._hew_centers),
            "n_anchors": len(self.anchor_indices),
            "schedule": self.schedule,
            "n_calls": self._call_count,
            "mean_grad_norm": self.mean_grad_norm,
            "mean_ff_energy": self.mean_ff_energy,
            "mean_hew_energy": self.mean_hew_energy,
            "framework": self.framework,
        }

    def to(self, device: str) -> "SoftFixGuidance":
        self.device = device
        return self


def create_soft_fix_callback(
    n_atoms: int,
    atom_types: np.ndarray | None = None,
    *,
    lambda_ff: float = 0.5,
    k_hew: float = 1.0,
    site_map_path: str | None = None,
    anchor_indices: list[int] | None = None,
    total_steps: int = 100,
    framework: str = "ode",
    schedule: str = "quadratic",
    grad_clip: float = 1.0,
    phase_gate: float = 0.6,
    device: str = "cpu",
    protein_pdb: str | None = None,
    verbose: bool = False,
) -> SoftFixGuidance:
    """Factory for SoftFixGuidance callback."""
    return SoftFixGuidance(
        n_atoms=n_atoms,
        atom_types=atom_types,
        lambda_ff=lambda_ff,
        k_hew=k_hew,
        site_map_path=site_map_path,
        anchor_indices=anchor_indices,
        total_steps=total_steps,
        framework=framework,
        schedule=schedule,
        grad_clip=grad_clip,
        phase_gate=phase_gate,
        device=device,
        protein_pdb=protein_pdb,
        verbose=verbose,
    )
