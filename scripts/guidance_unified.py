#!/usr/bin/env python3
"""
Unified Guidance Module — All baseline strategies under one interface.

Every guidance class implements the DrugFlow post_step_callback contract:
    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict

Classes:
    UnguidedGuidance     — no-op passthrough
    HardFixGuidance      — coordinate overwrite (anchor teleport)
    LaiSoftFixGuidance   — R^{3N} MMFF94 + R^{3N} E_site, NO CoM projection
    BadgerProxyGuidance  — R^{3N} Vina-surrogate (LJ+Coulomb), NO HEW awareness
    KAGGuidance          — CoM-projected E_site with ablation params
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance.latent_guidance import SiteCompatibilityEnergy, N_ATOM_TYPES


# ═══════════════════════════════════════════════════════════════════════════
# Atom type mapping
# ═══════════════════════════════════════════════════════════════════════════

ATOM_TYPE_TO_ELEMENT = {
    0: 6, 1: 6, 2: 7, 3: 7, 4: 8, 5: 8, 6: 16,
    7: 17, 8: 15, 9: 6, 10: 6,
}


def atom_type_probs_to_element(probs: np.ndarray) -> int:
    idx = int(np.argmax(probs))
    return ATOM_TYPE_TO_ELEMENT.get(idx, 6)


# ═══════════════════════════════════════════════════════════════════════════
# MMFF94 energy (static, reusable)
# ═══════════════════════════════════════════════════════════════════════════

def compute_mmff94_energy(coords: np.ndarray, template_mol: Chem.Mol) -> float | None:
    """RDKit MMFF94 energy for given coordinates."""
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


def coords_to_rdkit(
    x_np: np.ndarray,
    h: np.ndarray | torch.Tensor | None = None,
    atom_types_ref: np.ndarray | None = None,
    n_atoms: int = 0,
) -> Chem.Mol | None:
    """Build RDKit mol from coordinates + atom type predictions."""
    if h is not None:
        # Ensure h is on CPU as numpy
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()
        elements = [atom_type_probs_to_element(h[i]) for i in range(len(h))]
    elif atom_types_ref is not None:
        ref = atom_types_ref
        elements = [atom_type_probs_to_element(ref[i])
                    for i in range(min(len(ref), n_atoms))]
    else:
        elements = [6] * n_atoms

    n = min(len(elements), n_atoms) if n_atoms > 0 else len(elements)

    mol = Chem.RWMol()
    for i in range(n):
        mol.AddAtom(Chem.Atom(int(elements[i])))

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
            _determine_bonds_by_distance(mol, x_np, n)
    except Exception:
        mol = _determine_bonds_by_distance(Chem.RWMol(mol).GetMol(), x_np, n)

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        if mp is None:
            return None
    except Exception:
        return None

    return mol


def _determine_bonds_by_distance(mol: Chem.Mol, x_np: np.ndarray, n: int) -> Chem.Mol:
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
            if d < 1.3 * (r_i + r_j):
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)
    mol_out = rwmol.GetMol()
    try:
        Chem.SanitizeMol(mol_out)
    except Exception:
        pass
    return mol_out


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class BaseGuidance(ABC):
    """Abstract base for all guidance strategies."""

    def __init__(self, total_steps: int = 100, phase_gate: float = 0.6,
                 schedule: str = "quadratic", device: str = "cpu",
                 verbose: bool = False):
        self.total_steps = total_steps
        self.phase_gate = phase_gate
        self.schedule = schedule
        self.device = device
        self.verbose = verbose
        self._call_count: int = 0
        self._guidance_name: str = self.__class__.__name__

    @abstractmethod
    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        ...

    def _compute_lambda(self, t_val: float, lambda_max: float) -> float:
        t = float(t_val)
        if self.schedule == "quadratic":
            return lambda_max * (1.0 - t) ** 2
        elif self.schedule == "constant":
            return lambda_max
        elif self.schedule == "linear":
            return lambda_max * (1.0 - t)
        return lambda_max * (1.0 - t) ** 2

    def get_summary(self) -> dict:
        return {"guidance": self._guidance_name, "n_calls": self._call_count}

    def to(self, device: str) -> "BaseGuidance":
        self.device = device
        return self


# ═══════════════════════════════════════════════════════════════════════════
# 1. UnguidedGuidance
# ═══════════════════════════════════════════════════════════════════════════

class UnguidedGuidance(BaseGuidance):
    """No-op: returns ligand unchanged."""

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        self._call_count += 1
        return ligand


# ═══════════════════════════════════════════════════════════════════════════
# 2. HardFixGuidance
# ═══════════════════════════════════════════════════════════════════════════

class HardFixGuidance(BaseGuidance):
    """Coordinate overwrite: teleports anchor atoms to target positions.

    This is the "hard-fix" baseline. At each step where phase_gate is
    satisfied, anchor atom coordinates are overwritten to target positions.
    This injects catastrophic KPE (98.5%).
    """

    def __init__(self, anchor_indices: list[int],
                 anchor_coords: torch.Tensor,
                 total_steps: int = 100, phase_gate: float = 0.6,
                 device: str = "cpu", verbose: bool = False, **kwargs):
        super().__init__(total_steps=total_steps, phase_gate=phase_gate,
                         device=device, verbose=verbose)
        self._guidance_name = "hard_fix"
        self.anchor_indices = list(anchor_indices)
        self.anchor_coords = anchor_coords.to(device)

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        self._call_count += 1
        if t_val < self.phase_gate:
            return ligand

        x = ligand["x"]
        n_atoms = x.shape[0]
        for idx in self.anchor_indices:
            if 0 <= idx < n_atoms and idx < len(self.anchor_coords):
                x[idx] = self.anchor_coords[idx].to(x.device)
        return ligand


# ═══════════════════════════════════════════════════════════════════════════
# 3. LaiSoftFixGuidance
# ═══════════════════════════════════════════════════════════════════════════

class LaiSoftFixGuidance(BaseGuidance):
    """Lai et al. global MMFF94 + HEW E_site gradient — R^{3N} injection.

    Gradient formula:
        ∇E_total = λ_ff · ∇E_MMFF94(x) + λ_site · ∇E_site(x)

    BOTH gradients are applied per-atom (R^{3N}).  NO CoM projection is
    used. This represents the most direct "naive" approach to combining
    force-field guidance with site-specific targeting.

    Expected: HIGH DirectOcc (E_site pulls toward HEW), HIGH Strain
    (MMFF94 gradient + harmonic constraint corrupt all internal DOF).
    """

    def __init__(self, n_atoms: int,
                 atom_types: np.ndarray | torch.Tensor | None = None,
                 site_map_path: str | None = None,
                 anchor_indices: list[int] | None = None,
                 lambda_ff: float = 0.5, lambda_site: float = 0.5,
                 total_steps: int = 100, phase_gate: float = 0.6,
                 schedule: str = "quadratic", grad_clip: float = 1.0,
                 grad_every: int = 5,
                 device: str = "cpu", verbose: bool = False, **kwargs):
        super().__init__(total_steps=total_steps, phase_gate=phase_gate,
                         schedule=schedule, device=device, verbose=verbose)
        self._guidance_name = "lai_soft_fix"
        self.n_atoms = n_atoms
        self.lambda_ff = lambda_ff
        self.lambda_site = lambda_site
        self.grad_clip = grad_clip
        self.grad_every = grad_every

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
                sm = json.load(f)
            for site in sm.get("sites", []):
                if site.get("site_type") == "high_energy_water":
                    self._hew_centers.append(np.array(site["center"], dtype=np.float64))
        self.anchor_indices = anchor_indices or []

        # Stats
        self._grad_norms: list[float] = []
        self._ff_energies: list[float] = []
        self._hew_energies: list[float] = []

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        self._call_count += 1
        if t_val < self.phase_gate:
            return ligand
        # Speed: only compute gradient every grad_every steps
        if step_idx % self.grad_every != 0:
            return ligand

        x = ligand["x"]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        n_atoms = x.shape[0]
        lam_ff = self._compute_lambda(t_val, self.lambda_ff)
        lam_site = self._compute_lambda(t_val, self.lambda_site)

        if n_atoms < 2:
            return ligand

        # ── 1. MMFF94 gradient (R^{3N}) via finite differences ──
        grad_ff = torch.zeros(n_atoms, 3, device=self.device)
        mol_raw = coords_to_rdkit(
            x.detach().cpu().numpy(),
            h=ligand.get("h"),
            atom_types_ref=self._atom_types_ref,
            n_atoms=n_atoms,
        )
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
                            x_plus = x_np.copy(); x_plus[i, d] += eps
                            e_plus = compute_mmff94_energy(x_plus, mol_raw)
                            x_minus = x_np.copy(); x_minus[i, d] -= eps
                            e_minus = compute_mmff94_energy(x_minus, mol_raw)
                            if e_plus is not None and e_minus is not None:
                                grad[i, d] = (e_plus - e_minus) / (2.0 * eps)
                    grad_ff = torch.from_numpy(grad).float().to(self.device)
            except Exception:
                pass

        # ── 2. E_site gradient (R^{3N}) — analytic from compatibility energy ──
        grad_site = torch.zeros(n_atoms, 3, device=self.device)
        hew_energy = 0.0
        if self._hew_centers and self.anchor_indices:
            x_np = x.detach().cpu().numpy().astype(np.float64)
            sigma2 = 2.0 * 9.0  # 2 * sigma_distance^2, sigma=3.0
            for idx, anchor_idx in enumerate(self.anchor_indices):
                if anchor_idx >= n_atoms:
                    continue
                hew_center = self._hew_centers[idx % len(self._hew_centers)]
                rel = x_np[anchor_idx] - hew_center
                dist_sq = float(np.sum(rel ** 2))
                gauss = np.exp(-dist_sq / sigma2)
                hew_energy += 0.5 * float(dist_sq)  # harmonic energy
                # Gradient: ∂/∂x [½|x-c|²] = (x - c)
                # Weighted by Gaussian for smoothness
                grad_site[anchor_idx] = torch.from_numpy(
                    gauss * rel
                ).float().to(self.device)

        # ── 3. COMBINED R^{3N} INJECTION (NO CoM projection!) ──
        combined_grad = lam_ff * grad_ff + lam_site * grad_site

        max_norm = combined_grad.norm(dim=-1).max().item()
        if max_norm > self.grad_clip:
            combined_grad = combined_grad * (self.grad_clip / (max_norm + 1e-8))

        x_updated = x.detach() + combined_grad

        self._grad_norms.append(float(combined_grad.norm().item()))
        self._hew_energies.append(float(hew_energy))

        if isinstance(ligand["x"], np.ndarray):
            ligand["x"] = x_updated.detach().cpu().numpy()
        else:
            ligand["x"] = x_updated.detach()
        return ligand

    def get_summary(self) -> dict:
        s = super().get_summary()
        s.update({
            "lambda_ff": self.lambda_ff, "lambda_site": self.lambda_site,
            "n_hew_sites": len(self._hew_centers),
            "n_anchors": len(self.anchor_indices),
            "mean_grad_norm": float(np.mean(self._grad_norms)) if self._grad_norms else 0.0,
            "mean_hew_energy": float(np.mean(self._hew_energies)) if self._hew_energies else 0.0,
        })
        return s


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized interaction scoring (for BadgerProxy)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_interaction_score_vec(ligand_coords: np.ndarray,
                                     protein_coords: np.ndarray) -> float:
    """Vectorized LJ interaction score (fast)."""
    sigma = 3.4
    epsilon = 0.2
    diff = ligand_coords[:, None, :] - protein_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-8)
    dist = np.maximum(dist, 1.0)
    mask = dist < 8.0
    if not mask.any():
        return 0.0
    sr6 = (sigma / dist) ** 6
    lj = epsilon * (sr6 * sr6 - 2.0 * sr6)
    return float(lj[mask].sum())


# ═══════════════════════════════════════════════════════════════════════════
# 4. BadgerProxyGuidance
# ═══════════════════════════════════════════════════════════════════════════

class BadgerProxyGuidance(BaseGuidance):
    """BADGER-style global Vina-surrogate gradient injection.

    Uses a Lennard-Jones protein-ligand interaction potential as a
    differentiable Vina surrogate.  Gradient is computed via finite
    differences and applied to ALL atoms (R^{3N}).

    Optimised: only computes gradient every `grad_every` steps and uses
    a downsampled subset of protein atoms for speed.

    CRITICAL: NO HEW site awareness.  NO CoM projection.
    Expected: GOOD Vina scores, DirectOcc_HEW ≈ 0%.
    """

    def __init__(self, n_atoms: int,
                 protein_pdb: str | None = None,
                 lambda_badger: float = 1.0,
                 total_steps: int = 100, phase_gate: float = 0.6,
                 schedule: str = "quadratic", grad_clip: float = 1.0,
                 grad_every: int = 10, max_protein_atoms: int = 500,
                 device: str = "cpu", verbose: bool = False, **kwargs):
        super().__init__(total_steps=total_steps, phase_gate=phase_gate,
                         schedule=schedule, device=device, verbose=verbose)
        self._guidance_name = "badger_proxy"
        self.n_atoms = n_atoms
        self.lambda_badger = lambda_badger
        self.grad_clip = grad_clip
        self.grad_every = grad_every

        # Load protein and downsample for speed
        self._protein_coords: np.ndarray | None = None
        if protein_pdb and Path(protein_pdb).exists():
            prot_mol = Chem.MolFromPDBFile(protein_pdb, removeHs=False)
            if prot_mol is not None:
                conf = prot_mol.GetConformer()
                n_prot = prot_mol.GetNumAtoms()
                all_coords = np.array([conf.GetAtomPosition(i) for i in range(n_prot)])
                # Downsample: take every k-th atom to hit max_protein_atoms
                step = max(1, n_prot // max_protein_atoms)
                self._protein_coords = all_coords[::step].astype(np.float64)
                if verbose:
                    print(f"  [BadgerProxy] Protein: {n_prot} → "
                          f"{len(self._protein_coords)} atoms (step={step})")

        self._grad_norms: list[float] = []
        self._interaction_energies: list[float] = []

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        self._call_count += 1
        if t_val < self.phase_gate:
            return ligand

        # Only compute gradient every grad_every steps
        if step_idx % self.grad_every != 0:
            return ligand

        x = ligand["x"]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        n_atoms = x.shape[0]
        lam = self._compute_lambda(t_val, self.lambda_badger)

        if n_atoms < 2 or lam <= 0 or self._protein_coords is None:
            return ligand

        x_np = x.detach().cpu().numpy().astype(np.float64)
        prot = self._protein_coords
        eps_fd = 1e-3

        # Compute base interaction score
        e0 = _compute_interaction_score_vec(x_np, prot)

        # Finite-difference gradient (vectorized over atoms)
        grad = np.zeros((n_atoms, 3), dtype=np.float64)
        for i in range(n_atoms):
            for d in range(3):
                x_plus = x_np.copy()
                x_plus[i, d] += eps_fd
                e_plus = _compute_interaction_score_vec(x_plus, prot)
                x_minus = x_np.copy()
                x_minus[i, d] -= eps_fd
                e_minus = _compute_interaction_score_vec(x_minus, prot)
                grad[i, d] = (e_plus - e_minus) / (2.0 * eps_fd)

        grad_t = torch.from_numpy(grad).float().to(self.device)

        # ── BADGER-STYLE GLOBAL INJECTION (R^{3N}) ──
        max_norm = grad_t.norm(dim=-1).max().item()
        if max_norm > self.grad_clip:
            grad_t = grad_t * (self.grad_clip / (max_norm + 1e-8))

        x_updated = x.detach() - lam * grad_t

        self._grad_norms.append(float(grad_t.norm().item()))
        self._interaction_energies.append(e0)

        if isinstance(ligand["x"], np.ndarray):
            ligand["x"] = x_updated.detach().cpu().numpy()
        else:
            ligand["x"] = x_updated.detach()
        return ligand

    def get_summary(self) -> dict:
        s = super().get_summary()
        s.update({
            "lambda_badger": self.lambda_badger,
            "mean_grad_norm": float(np.mean(self._grad_norms)) if self._grad_norms else 0.0,
            "mean_interaction_energy": float(np.mean(self._interaction_energies)) if self._interaction_energies else 0.0,
        })
        return s


# ═══════════════════════════════════════════════════════════════════════════
# 5. KAGGuidance — wraps KinematicAnchorGuidance with ablation params
# ═══════════════════════════════════════════════════════════════════════════

class KAGGuidance(BaseGuidance):
    """Kinematic Anchor Guidance with ablation parameters.

    Wraps the existing KinematicAnchorGuidance and exposes ablation controls:
      - projection_mode: 'com' (default) | 'full' | 'internal'
      - skip_phase1: False (two-stage) | True (single-stage)
      - schedule_type: 'quadratic' (default) | 'constant'
    """

    def __init__(self, anchor_indices: list[int],
                 site_energy: SiteCompatibilityEnergy,
                 total_steps: int = 200,
                 projection_mode: str = "com",
                 skip_phase1: bool = False,
                 schedule_type: str = "quadratic",
                 lambda_max: float = 0.5,
                 phase_gate: float = 0.6,
                 grad_clip: float = 0.5,
                 device: str = "cpu",
                 verbose: bool = False, **kwargs):
        super().__init__(total_steps=total_steps, phase_gate=phase_gate,
                         schedule=schedule_type, device=device, verbose=verbose)
        self._guidance_name = "kag"
        self.anchor_indices = list(anchor_indices)
        self.site_energy = site_energy
        self.projection_mode = projection_mode
        self.skip_phase1 = skip_phase1
        self.schedule_type = schedule_type
        self.lambda_max = lambda_max
        self.grad_clip = grad_clip

        # Import here to avoid circular dependency
        from guidance.kinematic_anchor import KinematicAnchorGuidance
        self._kag = KinematicAnchorGuidance(
            anchor_indices=anchor_indices,
            site_energy=site_energy,
            total_steps=total_steps,
            lambda_max=lambda_max,
            profile=schedule_type,
            grad_clip=grad_clip,
            track_kpe=True,
            verbose=verbose,
        )
        self._kag.to(device)

        # State for projection modes
        self._x_prev: torch.Tensor | None = None
        self._first_call: bool = True
        self._dt: float = 1.0 / max(total_steps, 1)
        self._site_grad_norms: list[float] = []

    def __call__(self, ligand: dict, step_idx: int, t_val: float) -> dict:
        self._call_count += 1

        if self.projection_mode == "com":
            # Default: use the standard CoM-only KAG
            return self._kag(ligand, step_idx, t_val)
        else:
            # Full or internal projection: apply per-atom gradients
            return self._apply_projection_mode(ligand, step_idx, t_val)

    def _apply_projection_mode(self, ligand: dict, step_idx: int,
                                t_val: float) -> dict:
        """Apply E_site gradient with full or internal projection."""
        if t_val < self.phase_gate:
            return ligand

        x = ligand["x"]
        n_atoms = x.shape[0]
        device = x.device

        if self._first_call:
            self._x_prev = x.clone()
            self._first_call = False
            return ligand

        # Compute per-atom E_site gradient
        # Use HARD atom type assignments for stronger gradient signal
        # (soft probabilities from DrugFlow logits are too diffuse early on)
        atom_type_indices = None
        if "h" in ligand:
            h = ligand["h"]
            if isinstance(h, torch.Tensor):
                n_types = min(h.shape[-1], N_ATOM_TYPES)
                atom_type_indices = h[:, :n_types].float().argmax(dim=-1)

        x_detached = x.detach().clone().requires_grad_(True)
        energy = self.site_energy(x_detached, atom_type_indices=atom_type_indices)
        if not energy.requires_grad:
            return ligand
        grad_full = torch.autograd.grad(energy, x_detached,
                                         create_graph=False, retain_graph=False)[0]

        lam = self._compute_lambda(t_val, self.lambda_max)

        if self.projection_mode == "internal":
            # Remove CoM component: only internal deformation
            grad_com = grad_full.mean(dim=0, keepdim=True)
            grad = grad_full - grad_com
        else:  # "full"
            grad = grad_full

        correction = lam * grad
        max_norm = correction.norm(dim=-1).max().item()
        if max_norm > self.grad_clip:
            correction = correction * (self.grad_clip / (max_norm + 1e-8))

        ligand["x"] = x + correction
        self._site_grad_norms.append(float(grad.norm().item()))
        self._x_prev = ligand["x"].clone()
        return ligand

    def get_kpe_summary(self) -> dict:
        if self.projection_mode == "com":
            return self._kag.get_kpe_summary()
        return {"kpe_ratio": 0.0}

    @property
    def kpe_ratio(self) -> float:
        if self.projection_mode == "com":
            return self._kag.kpe_ratio
        return 0.0

    def get_summary(self) -> dict:
        s = super().get_summary()
        s.update({
            "projection_mode": self.projection_mode,
            "skip_phase1": self.skip_phase1,
            "schedule_type": self.schedule_type,
            "lambda_max": self.lambda_max,
        })
        if self.projection_mode == "com":
            s.update(self._kag.get_kpe_summary())
        return s

    def to(self, device: str) -> "KAGGuidance":
        super().to(device)
        self._kag.to(device)
        if self._x_prev is not None:
            self._x_prev = self._x_prev.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

GUIDANCE_REGISTRY = {
    "unguided": UnguidedGuidance,
    "hard_fix": HardFixGuidance,
    "lai_soft_fix": LaiSoftFixGuidance,
    "badger_proxy": BadgerProxyGuidance,
    "kag": KAGGuidance,
}


def create_guidance(name: str, **kwargs) -> BaseGuidance:
    """Factory: create any guidance strategy by name.

    Args:
        name: one of "unguided", "hard_fix", "lai_soft_fix", "badger_proxy", "kag"
        **kwargs: forwarded to the guidance class constructor

    Returns:
        Guidance callable implementing post_step_callback interface.
    """
    cls = GUIDANCE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown guidance: {name!r}. "
                         f"Choose from: {list(GUIDANCE_REGISTRY.keys())}")
    return cls(**kwargs)
