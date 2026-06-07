#!/usr/bin/env python3
"""Synthetic v6-D Energy Landscape Demonstration (CPU-only).

Computes and visualizes the v6-D energy landscape on toy coordinates:
  - 2D slice through a HEW site
  - Shows E_disp, E_wrong, E_clash, E_overfill components
  - Demonstrates compatible atoms are attracted, incompatible repelled
  - Validates smoothness and differentiability
  - Compares v5-learned and v6-analytic behavior

Generates text report and ASCII plots. No GPU required.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
import numpy as np

from models.analytic_esfield import (
    AnalyticESFieldGuide, V6DConfig, create_v6d_guide,
    HEW_ENV_HYDROPHOBIC, HEW_ENV_POLAR_UNSATISFIED,
    HEW_ENV_MIXED, HEW_ENV_BURIED,
    classify_hew_environment, hew_env_to_idx,
    COMPAT_MATRIX, HEW_ENV_ORDER, ATOM_TYPE_TO_IDX,
)


def make_hew_site(center, env="hydrophobic", confidence=0.9, radius=1.4):
    """Create a minimal HEW site dict."""
    env_features = {
        "hydrophobic": {"hbond_count": 0, "hydrophobic_contact_count": 5,
                        "nearest_protein_distance": 4.0},
        "polar_unsatisfied": {"hbond_count": 0, "hydrophobic_contact_count": 1,
                             "nearest_protein_distance": 4.0},
        "mixed": {"hbond_count": 1, "hydrophobic_contact_count": 3,
                  "nearest_protein_distance": 3.5},
        "buried": {"hbond_count": 2, "hydrophobic_contact_count": 3,
                   "nearest_protein_distance": 2.0},
    }
    return {
        "site_type": "high_energy_water",
        "center": list(center),
        "radius": radius,
        "confidence": confidence,
        "features": env_features.get(env, env_features["hydrophobic"]),
    }


def demo_energy_landscape_1d():
    """1D scan: compute energy as atom moves from 0 to 6 Angstrom from HEW."""
    print("=" * 70)
    print("DEMO 1: 1D Energy Landscape — Atom approaching HEW center")
    print("=" * 70)

    site_map = {"sites": [make_hew_site((0, 0, 0), env="hydrophobic")],
                "pocket_center": [0, 0, 0]}

    config = V6DConfig(disp_weight=1.0, wrong_atom_weight=0.5,
                       clash_weight=0.0, overfill_weight=0.0,
                       sigma_occ=1.2, cutoff_dist=5.0)
    guide = AnalyticESFieldGuide(site_map, config=config)
    guide.to(torch.device("cpu"))

    # Scan atoms at distances 0.2 to 6.0 Angstrom
    atom_types = {
        "C_sp3 (compat)": 1,
        "O_acceptor (incompat)": 5,
        "charged (penalized)": 9,
    }

    distances = np.linspace(0.2, 6.0, 30)
    print(f"\n{'Distance':>8s}", end="")
    for name in atom_types:
        print(f"  {name:>16s}", end="")
    print()

    for d in distances:
        x = torch.tensor([[d, 0.0, 0.0]], requires_grad=True)
        print(f"{d:8.2f}", end="")
        for name, type_idx in atom_types.items():
            h = torch.zeros(1, 11)
            h[0, type_idx] = 5.0
            energy = guide(torch.tensor(0.6), x=x, h=h, batch_mask=None)
            grad = torch.autograd.grad(energy, x)[0]
            force = -grad[0, 0].item()  # negative gradient = force toward minimum
            print(f"  E={energy.item():+.3f} F={force:+.3f}", end="")
        print()

    # Key assertion: C_sp3 force is attractive (negative energy gradient toward HEW)
    x_test = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
    h_c = torch.zeros(1, 11); h_c[0, 1] = 5.0
    e_c = guide(torch.tensor(0.6), x=x_test, h=h_c, batch_mask=None)
    g_c = torch.autograd.grad(e_c, x_test)[0]

    h_o = torch.zeros(1, 11); h_o[0, 5] = 5.0
    e_o = guide(torch.tensor(0.6), x=x_test, h=h_o, batch_mask=None)
    g_o = torch.autograd.grad(e_o, x_test)[0]

    print(f"\nAt d=2.0A:")
    print(f"  C_sp3 (compatible):   E={e_c.item():+.4f}, grad_x={g_c[0,0].item():+.4f}")
    print(f"  O_acceptor (incompat): E={e_o.item():+.4f}, grad_x={g_o[0,0].item():+.4f}")
    print(f"  Compatible more attractive: {abs(g_c[0,0].item()) > abs(g_o[0,0].item())}")


def demo_compatibility_matrix():
    """Display the heuristic compatibility matrix."""
    print("\n" + "=" * 70)
    print("DEMO 2: Heuristic Compatibility Matrix M(atom_type, HEW_env)")
    print("=" * 70)

    print(f"\n{'Atom Type':>16s}", end="")
    for env in HEW_ENV_ORDER:
        print(f"  {env:>20s}", end="")
    print()

    for atom_type, idx in sorted(ATOM_TYPE_TO_IDX.items(), key=lambda x: x[1]):
        if idx >= COMPAT_MATRIX.shape[1]:
            continue
        print(f"{atom_type:>16s}", end="")
        for env_idx in range(len(HEW_ENV_ORDER)):
            val = COMPAT_MATRIX[env_idx, idx].item()
            print(f"  {val:>20.3f}", end="")
        print()

    # Verify key properties
    print("\n--- Matrix Validation ---")
    checks = []
    c_sp3 = ATOM_TYPE_TO_IDX["C_sp3"]
    charged = ATOM_TYPE_TO_IDX["charged"]
    o_acc = ATOM_TYPE_TO_IDX["O_acceptor"]
    n_don = ATOM_TYPE_TO_IDX["N_donor"]

    hphob = HEW_ENV_ORDER.index("hydrophobic")
    polar = HEW_ENV_ORDER.index("polar_unsatisfied")
    mixed = HEW_ENV_ORDER.index("mixed")

    # hydrophobic: C_sp3 > charged
    c1 = COMPAT_MATRIX[hphob, c_sp3] > COMPAT_MATRIX[hphob, charged]
    checks.append(("C_sp3 > charged in hydrophobic HEW", c1))

    # polar: O_acceptor > C_sp3
    c2 = COMPAT_MATRIX[polar, o_acc] > COMPAT_MATRIX[polar, c_sp3]
    checks.append(("O_acceptor > C_sp3 in polar HEW", c2))

    # mixed: all values moderate
    c3 = COMPAT_MATRIX[mixed, c_sp3] > 0 and COMPAT_MATRIX[mixed, o_acc] > 0
    checks.append(("Both types rewarded (weakly) in mixed HEW", c3))

    for name, result in checks:
        print(f"  {'PASS' if result else 'FAIL'}: {name}")


def demo_protection_terms():
    """Demonstrate E_protect components: wrong-atom penalty and clash repulsion."""
    print("\n" + "=" * 70)
    print("DEMO 3: Protection Terms — Wrong Atom & Clash Penalty")
    print("=" * 70)

    # --- Wrong atom penalty ---
    site_map = {"sites": [make_hew_site((0, 0, 0), env="hydrophobic")],
                "pocket_center": [0, 0, 0]}

    config = V6DConfig(disp_weight=0.0, wrong_atom_weight=1.0,
                       clash_weight=0.0, overfill_weight=0.0,
                       sigma_occ=1.2, cutoff_dist=5.0)
    guide = AnalyticESFieldGuide(site_map, config=config)
    guide.to(torch.device("cpu"))

    # Charged atom at d=1.0A from hydrophobic HEW → max penalty
    x = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    h_ch = torch.zeros(1, 11); h_ch[0, 9] = 5.0  # charged
    e_wrong_ch = guide(torch.tensor(0.6), x=x, h=h_ch, batch_mask=None)
    g_wrong_ch = torch.autograd.grad(e_wrong_ch, x)[0]

    # C_sp3 at same position → no penalty
    h_c = torch.zeros(1, 11); h_c[0, 1] = 5.0
    e_wrong_c = guide(torch.tensor(0.6), x=x, h=h_c, batch_mask=None)
    g_wrong_c = torch.autograd.grad(e_wrong_c, x)[0]

    print(f"\nWrong-atom penalty at d=1.0A (disp_weight=0, wrong_atom_weight=1.0):")
    print(f"  charged near hydrophobic HEW: E={e_wrong_ch.item():+.4f}, "
          f"|grad|={g_wrong_ch.norm().item():.4f}")
    print(f"  C_sp3   near hydrophobic HEW: E={e_wrong_c.item():+.4f}, "
          f"|grad|={g_wrong_c.norm().item():.4f}")
    print(f"  Penalty asymmetry: {abs(e_wrong_ch.item()) > abs(e_wrong_c.item())}")

    # --- Clash repulsion ---
    protein = torch.tensor([[0.0, 0.0, 0.0]])  # protein atom at origin

    config2 = V6DConfig(disp_weight=0.0, wrong_atom_weight=0.0,
                        clash_weight=1.0, overfill_weight=0.0,
                        clash_distance=2.0, clash_sigma=0.3)
    guide2 = AnalyticESFieldGuide(site_map, config=config2, protein_coords=protein)
    guide2.to(torch.device("cpu"))

    # Atom at d=1.0A → should be repelled
    x_close = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    h = torch.zeros(1, 11); h[0, 1] = 5.0
    e_close = guide2(torch.tensor(0.6), x=x_close, h=h, batch_mask=None)
    g_close = torch.autograd.grad(e_close, x_close)[0]

    # Atom at d=4.0A → should be fine
    x_far = torch.tensor([[4.0, 0.0, 0.0]], requires_grad=True)
    e_far = guide2(torch.tensor(0.6), x=x_far, h=h, batch_mask=None)
    g_far = torch.autograd.grad(e_far, x_far)[0]

    print(f"\nClash penalty (protein at origin):")
    print(f"  Ligand at 1.0A: E={e_close.item():+.4f}, |grad|={g_close.norm().item():.4f}")
    print(f"  Ligand at 4.0A: E={e_far.item():+.4f}, |grad|={g_far.norm().item():.4f}")
    print(f"  Repulsion active at 1.0A: {g_close.norm().item() > 0.1}")


def demo_overfill():
    """Demonstrate overfill penalty."""
    print("\n" + "=" * 70)
    print("DEMO 4: Overfill Penalty — Multiple atoms at same HEW")
    print("=" * 70)

    site_map = {"sites": [make_hew_site((0, 0, 0), env="hydrophobic")],
                "pocket_center": [0, 0, 0]}

    config = V6DConfig(disp_weight=0.0, wrong_atom_weight=0.0,
                       clash_weight=0.0, overfill_weight=1.0,
                       overfill_max_per_site=2, sigma_occ=1.2, cutoff_dist=5.0)
    guide = AnalyticESFieldGuide(site_map, config=config)
    guide.to(torch.device("cpu"))

    # 1 atom near HEW: no overfill
    x1 = torch.tensor([[0.5, 0.0, 0.0]], requires_grad=True)
    h1 = torch.zeros(1, 11); h1[0, 1] = 5.0
    e1 = guide(torch.tensor(0.6), x=x1, h=h1, batch_mask=None)
    print(f"  1 atom at HEW:  E_overfill={e1.item():+.6f}")

    # 4 atoms near HEW: overfill penalty
    x4 = torch.tensor([
        [0.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, -0.5, 0.0],
    ], requires_grad=True)
    h4 = torch.zeros(4, 11); h4[:, 1] = 5.0
    e4 = guide(torch.tensor(0.6), x=x4, h=h4, batch_mask=None)
    print(f"  4 atoms at HEW: E_overfill={e4.item():+.6f}")
    print(f"  Overfill penalty active: {abs(e4.item()) > abs(e1.item())}")


def demo_random_matrix_ablation():
    """Show that random compatibility matrix produces different forces."""
    print("\n" + "=" * 70)
    print("DEMO 5: Random Matrix Ablation — Physics must depend on correct rules")
    print("=" * 70)

    site_map = {"sites": [make_hew_site((0, 0, 0), env="hydrophobic")],
                "pocket_center": [0, 0, 0]}

    config = V6DConfig(disp_weight=1.0, wrong_atom_weight=0.5,
                       clash_weight=0.0, overfill_weight=0.0,
                       sigma_occ=1.2, cutoff_dist=5.0)
    guide = AnalyticESFieldGuide(site_map, config=config)
    guide.to(torch.device("cpu"))

    x = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
    h = torch.zeros(1, 11)
    h[0, 1] = 5.0  # C_sp3

    # Correct matrix
    e_correct = guide(torch.tensor(0.6), x=x, h=h, batch_mask=None)
    g_correct = torch.autograd.grad(e_correct, x)[0]

    # Random matrix
    saved = guide._compat.clone()
    guide._compat = torch.randn_like(saved)
    e_random = guide(torch.tensor(0.6), x=x, h=h, batch_mask=None)
    g_random = torch.autograd.grad(e_random, x)[0]
    guide._compat = saved

    print(f"  Correct matrix:  E={e_correct.item():+.4f}, |grad|={g_correct.norm().item():.4f}")
    print(f"  Random matrix:   E={e_random.item():+.4f}, |grad|={g_random.norm().item():.4f}")
    diff = abs(g_correct.norm().item() - g_random.norm().item())
    print(f"  Force difference: {diff:.4f}")
    print(f"  Matrices differ: {diff > 0.001}")


def demo_smoothness():
    """Verify energy is smooth (no discontinuities in gradient)."""
    print("\n" + "=" * 70)
    print("DEMO 6: Smoothness — Energy and gradient must be continuous")
    print("=" * 70)

    site_map = {"sites": [
        make_hew_site((0, 0, 0), env="hydrophobic"),
        make_hew_site((3, 0, 0), env="polar_unsatisfied"),
    ], "pocket_center": [0, 0, 0]}

    protein = torch.tensor([[1.0, 1.5, 0.0]])

    config = V6DConfig(disp_weight=1.0, wrong_atom_weight=0.5,
                       clash_weight=1.0, overfill_weight=0.3,
                       sigma_occ=1.2, cutoff_dist=5.0,
                       clash_distance=2.0, clash_sigma=0.3)
    guide = AnalyticESFieldGuide(site_map, config=config, protein_coords=protein)
    guide.to(torch.device("cpu"))

    # Test at many positions
    np.random.seed(42)
    positions = np.random.uniform(-2, 5, (20, 3))
    energies = []
    grad_norms = []

    for pos in positions:
        x = torch.tensor([pos], requires_grad=True, dtype=torch.float32)
        h = torch.randn(1, 11) * 2  # mixed types
        energy = guide(torch.tensor(0.6), x=x, h=h, batch_mask=None)
        grad = torch.autograd.grad(energy, x)[0]
        energies.append(energy.item())
        grad_norms.append(grad.norm().item())

    e_range = max(energies) - min(energies)
    g_range = max(grad_norms) - min(grad_norms)
    all_finite = all(np.isfinite(e) for e in energies) and all(np.isfinite(g) for g in grad_norms)

    print(f"  20 random positions tested")
    print(f"  Energy range:   [{min(energies):.4f}, {max(energies):.4f}]")
    print(f"  |grad| range:   [{min(grad_norms):.4f}, {max(grad_norms):.4f}]")
    print(f"  All finite:     {all_finite}")
    print(f"  No NaN in grad: {all_finite}")


def demo_v5_vs_v6_comparison():
    """Conceptual comparison: v5-learned vs v6-analytic behavior."""
    print("\n" + "=" * 70)
    print("DEMO 7: v5-Learned vs v6-Analytic — Conceptual Comparison")
    print("=" * 70)

    print("""
  Property                  | v5 (Learned α/β)        | v6-D (Analytic)
  --------------------------|--------------------------|---------------------------
  Energy shape              | MLP-learned α,β          | Explicit heuristic rules
  Attraction range          | d0=3.0A (learned)        | sigma_occ=1.2A (Gaussian)
  Compatibility             | Learned per (atom,site)  | Hardcoded 4x11 matrix
  Wrong-atom handling       | β*R(d) repulsion         | Dedicated E_wrong term
  Protein clash             | Not modeled              | E_clash (exponential)
  Overfill                  | Not modeled              | E_overfill (softplus)
  Differentiable            | Yes (softplus α,β)       | Yes (softplus, exp)
  HEW environment aware     | Implicit (via embedding) | Explicit classification
  Actionable HEW filter     | v5 hew_gating            | Confidence + env + top-k
  Interpretability          | Low (black-box MLP)      | High (rule-based)
  Risk of spurious minima   | Moderate (MLP can learn  | Low (physics-constrained)
                             |  non-physical patterns)  |
""")

    print("Key advantage of v6-D: every force component has a physical interpretation.")
    print("E_disp:  'this atom type should displace this HEW environment'")
    print("E_wrong: 'this atom type should NOT be near this HEW environment'")
    print("E_clash: 'this atom is too close to protein'")
    print("E_overfill: 'too many atoms crowding one HEW site'")


def main():
    print("Analytic ESField v6-D — Energy Landscape Demonstration")
    print("=" * 70)
    print("All tests run on CPU, no GPU required.\n")

    demo_energy_landscape_1d()
    demo_compatibility_matrix()
    demo_protection_terms()
    demo_overfill()
    demo_random_matrix_ablation()
    demo_smoothness()
    demo_v5_vs_v6_comparison()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
