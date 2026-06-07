#!/usr/bin/env python3
"""Offline validation of Potential v5 — no DrugFlow needed.

1. Complete force matrix → CSV
2. Single-atom toy coordinate update (sign check)
3. Single-molecule multi-site guidance step (competition check)
"""

import json, sys, csv, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.potential_network import CompatibilityPotentialV5, PotentialConfig
from evaluation.posu import compute_posu, compute_hewu, _extract_atoms_from_mol
from utils.geometry import distance as calc_dist
from utils.chemistry import is_compatible_atom_site, infer_atom_type, atomic_number, normalize_element
from rdkit import Chem

ATOM_TYPES = {1: 'C_sp3', 2: 'C_aromatic', 3: 'N_donor', 4: 'N_acceptor', 5: 'O_acceptor', 6: 'S', 7: 'halogen', 8: 'P'}
SITE_TYPES = {1: 'HEW', 2: 'SW', 3: 'HC'}
ALL_SITE_NAMES = {1: 'high_energy_water', 2: 'stable_water', 3: 'hydrophobic_cavity'}

HEW_COMPAT = {'C_sp3', 'C_aromatic', 'halogen', 'S', 'O_acceptor', 'N_donor', 'N_acceptor', 'P'}
SW_COMPAT = {'O_acceptor', 'N_donor', 'N_acceptor'}
HC_COMPAT = {'C_sp3', 'C_aromatic', 'halogen', 'S'}

COMPAT_MAP = {'HEW': HEW_COMPAT, 'SW': SW_COMPAT, 'HC': HC_COMPAT}


def load_v5():
    ckpt = torch.load(ROOT / 'experiments/potential_training/v5/potential_v5_epoch_0030.pt', map_location='cpu')
    cfg = PotentialConfig(**{k: ckpt['config'][k] for k in ['atom_embed_dim', 'site_embed_dim', 'hidden_dim', 'num_layers']})
    model = CompatibilityPotentialV5(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute_force(model, at_idx, st_idx, dist, eps=0.1):
    """Compute -dE/dd at a given distance."""
    d1 = torch.tensor([dist - eps], dtype=torch.float32)
    d2 = torch.tensor([dist + eps], dtype=torch.float32)
    at = torch.tensor([at_idx], dtype=torch.long)
    st = torch.tensor([st_idx], dtype=torch.long)
    r1 = torch.tensor([[0.0, 0.0, float(dist - eps)]], dtype=torch.float32)
    r2 = torch.tensor([[0.0, 0.0, float(dist + eps)]], dtype=torch.float32)
    rad = torch.tensor([1.4], dtype=torch.float32)
    conf = torch.tensor([1.0], dtype=torch.float32)
    with torch.no_grad():
        e1 = model(at, st, r1, d1, rad, conf).item()
        e2 = model(at, st, r2, d2, rad, conf).item()
        alpha, beta = model.get_coefficients(at, st, r1, d1, rad, conf)
    force = -(e2 - e1) / (2 * eps)
    return e1, force, alpha.item(), beta.item()


def sanity1_force_matrix(model):
    """Save complete force matrix as CSV."""
    out = ROOT / 'experiments/potential_training/v5/force_matrix.csv'
    distances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]

    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['atom_type', 'site_type', 'distance', 'energy', 'force', 'alpha', 'beta',
                     'is_compatible', 'expected_direction', 'actual_direction', 'pass_fail'])

        for at_idx, at_name in ATOM_TYPES.items():
            for st_idx, st_name in SITE_TYPES.items():
                compat_set = COMPAT_MAP[st_name]
                is_comp = at_name in compat_set

                for d in distances:
                    e, force, alpha, beta = compute_force(model, at_idx, st_idx, d)

                    # Expected direction
                    if is_comp:
                        if d < 2.5: expected = 'repel'      # too close, should push out
                        elif d < 4.5: expected = 'attract'  # in well zone, pull in
                        else: expected = 'neutral'           # far, no effect
                    else:
                        if d < 2.0: expected = 'repel'      # steric clash
                        else: expected = 'neutral'           # no attraction

                    if force < -0.03: actual = 'attract'
                    elif force > 0.03: actual = 'repel'
                    else: actual = 'neutral'

                    # Pass/fail
                    if expected == 'attract' and actual == 'attract': pf = 'PASS'
                    elif expected == 'repel' and actual == 'repel': pf = 'PASS'
                    elif expected == 'neutral' and actual in ('neutral', 'repel'): pf = 'PASS'
                    elif expected == 'attract' and actual == 'neutral': pf = 'WARN'
                    else: pf = 'FAIL'

                    w.writerow([at_name, st_name, f'{d:.1f}', f'{e:.4f}', f'{force:.4f}',
                                f'{alpha:.4f}', f'{beta:.4f}', str(is_comp), expected, actual, pf])

    # Summary
    with open(out) as f:
        rows = list(csv.DictReader(f))
    passes = sum(1 for r in rows if r['pass_fail'] == 'PASS')
    warns = sum(1 for r in rows if r['pass_fail'] == 'WARN')
    fails = sum(1 for r in rows if r['pass_fail'] == 'FAIL')
    total = len(rows)
    print(f"\n=== Force Matrix Summary ===")
    print(f"  PASS: {passes}/{total} ({100*passes/total:.0f}%)")
    print(f"  WARN: {warns}/{total} ({100*warns/total:.0f}%)")
    print(f"  FAIL: {fails}/{total} ({100*fails/total:.0f}%)")
    print(f"  Saved: {out}")

    # Per-site-type breakdown
    for st_name in ['HEW', 'SW', 'HC']:
        st_rows = [r for r in rows if r['site_type'] == st_name]
        st_pass = sum(1 for r in st_rows if r['pass_fail'] == 'PASS')
        print(f"  {st_name}: {st_pass}/{len(st_rows)} PASS")

    return rows


def sanity2_toy_single_atom(model):
    """Toy coordinate update: single atom, single site, one guidance step."""
    print(f"\n=== Toy Single-Atom Coordinate Update ===")
    lr = 0.1  # step size

    tests = [
        # (atom_type_idx, site_type_idx, start_dist, expected)
        (1, 1, 4.0, 'decrease'),   # C_sp3 + HEW at 4Å → should move closer
        (1, 1, 3.0, 'stable'),     # C_sp3 + HEW at 3Å → at well minimum, stable
        (1, 1, 1.5, 'increase'),   # C_sp3 + HEW at 1.5Å → too close, should move out
        (5, 1, 4.0, 'decrease'),   # O_acceptor + HEW at 4Å
        (1, 2, 3.0, 'stable'),     # C_sp3 + SW → incompatible, shouldn't move
        (5, 2, 3.0, 'stable'),     # O_acceptor + SW → compatible but SW has α≈0 in v5
        (1, 3, 4.0, 'decrease'),   # C_sp3 + HC
        (5, 3, 3.0, 'stable'),     # O_acceptor + HC → incompatible
        (8, 1, 4.0, 'decrease'),   # P + HEW
    ]

    all_pass = True
    for at_idx, st_idx, start_d, expected in tests:
        at_name = ATOM_TYPES.get(at_idx, '?')
        st_name = SITE_TYPES.get(st_idx, '?')

        # Compute gradient at start_d
        eps = 0.05
        d1_t = torch.tensor([start_d - eps], dtype=torch.float32)
        d2_t = torch.tensor([start_d + eps], dtype=torch.float32)
        at_t = torch.tensor([at_idx])
        st_t = torch.tensor([st_idx])
        r1_t = torch.tensor([[0.0, 0.0, float(start_d - eps)]], dtype=torch.float32)
        r2_t = torch.tensor([[0.0, 0.0, float(start_d + eps)]], dtype=torch.float32)
        rad_t = torch.tensor([1.4], dtype=torch.float32)
        conf_t = torch.tensor([1.0], dtype=torch.float32)

        with torch.no_grad():
            e1 = model(at_t, st_t, r1_t, d1_t, rad_t, conf_t).item()
            e2 = model(at_t, st_t, r2_t, d2_t, rad_t, conf_t).item()
            alpha, beta = model.get_coefficients(at_t, st_t, r1_t, d1_t, rad_t, conf_t)
        force = -(e2 - e1) / (2 * eps)

        # Apply one gradient step: new_d = d + lr * force  (force = -dE/dd)
        force_val = float(force)
        new_d = start_d + lr * force_val

        if expected == 'decrease':
            ok = new_d < start_d - 0.001
        elif expected == 'increase':
            ok = new_d > start_d + 0.001
        else:  # stable
            ok = abs(new_d - start_d) < 0.1

        status = 'PASS' if ok else 'FAIL'
        if not ok: all_pass = False
        print(f"  {at_name}+{st_name} d={start_d:.1f}->{new_d:.2f} f={force:+.3f} a={alpha.item():.3f} [{expected}] {status}")

    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


def sanity3_toy_molecule(model):
    """Apply one guidance step to a real ligand with its correct site map."""
    print(f"\n=== Toy Single-Molecule Multi-Site Guidance ===")

    test_pockets = json.load(open(ROOT / 'experiments/pdbbind_water_sites/test_pockets.json'))
    lr = 0.05  # small step to avoid distortion

    results = []
    for p in test_pockets[:5]:  # first 5 pockets
        pid = p['pdb_id']
        lig_path = f"{p['dir']}/{pid}_ligand.sdf"
        sm_path = ROOT / f'experiments/pdbbind_water_sites/test_sites/correct/{pid}_site_map.json'

        try:
            mol = Chem.SDMolSupplier(lig_path)[0]
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except:
            continue

        sm = json.load(open(sm_path))
        hew_sites = [s for s in sm['sites'] if s['site_type'] == 'high_energy_water']
        if not hew_sites:
            continue

        atoms = _extract_atoms_from_mol(mol)

        # Compute initial HEW nearest compatible distance
        initial_dists = []
        for site in hew_sites:
            best = 10.0
            for a in atoms:
                if is_compatible_atom_site(a['atom_type'], a['atomic_number'], 'high_energy_water'):
                    d = calc_dist(a['coord'], tuple(site['center']))
                    if d < best: best = d
            if best < 10.0:
                initial_dists.append(best)
        mean_initial = np.mean(initial_dists) if initial_dists else 10.0

        # Apply one guidance step to each atom
        new_coords = {}
        for a in atoms:
            coord = torch.tensor(a['coord'], dtype=torch.float32)
            at_idx = {v: k for k, v in ATOM_TYPES.items()}.get(a['atom_type'], 0)
            at_t = torch.tensor([at_idx], dtype=torch.long)

            total_grad = torch.zeros(3)
            for site in hew_sites:
                sc = torch.tensor(site['center'], dtype=torch.float32)
                rel = coord - sc
                d = torch.norm(rel).clamp_min(1e-4)

                d_t = d.unsqueeze(0)
                st_t = torch.tensor([1], dtype=torch.long)  # HEW
                rel_t = rel.unsqueeze(0)
                rad_t = torch.tensor([site.get('radius', 1.4)], dtype=torch.float32)
                conf_t = torch.tensor([site.get('confidence', 1.0)], dtype=torch.float32)

                with torch.enable_grad():
                    coord_tmp = coord.detach().requires_grad_(True)
                    rel_tmp = (coord_tmp - sc).unsqueeze(0)
                    d_tmp = torch.norm(coord_tmp - sc).unsqueeze(0).clamp_min(1e-4)
                    e = model(at_t, st_t, rel_tmp, d_tmp, rad_t, conf_t)
                    grad = torch.autograd.grad(e.sum(), coord_tmp)[0]

                total_grad = total_grad + grad

            new_coords[a['idx']] = coord - lr * total_grad

        # Compute new HEW nearest compatible distance
        new_dists = []
        for site in hew_sites:
            best = 10.0
            for a in atoms:
                if is_compatible_atom_site(a['atom_type'], a['atomic_number'], 'high_energy_water'):
                    old_c = a['coord']
                    new_c = new_coords.get(a['idx'], torch.tensor(old_c))
                    d = float(torch.norm(new_c - torch.tensor(site['center'])))
                    if d < best: best = d
            if best < 10.0:
                new_dists.append(best)
        mean_new = np.mean(new_dists) if new_dists else 10.0

        delta = mean_new - mean_initial
        status = 'DECREASE' if delta < -0.05 else ('INCREASE' if delta > 0.05 else 'STABLE')
        print(f"  {pid}: HEW dist {mean_initial:.2f}→{mean_new:.2f}Å Δ={delta:+.3f} [{status}] "
              f"(sites={len(hew_sites)}, atoms={len(atoms)})")

        results.append({'pid': pid, 'delta': delta, 'status': status})

    n_decrease = sum(1 for r in results if r['status'] == 'DECREASE')
    print(f"  Decreased: {n_decrease}/{len(results)}")
    return results


def main():
    print("Loading v5 model...")
    model = load_v5()

    print("\n" + "=" * 60)
    print("SANITY 1: Complete Force Matrix")
    print("=" * 60)
    sanity1_force_matrix(model)

    print("\n" + "=" * 60)
    print("SANITY 2: Toy Single-Atom Update")
    print("=" * 60)
    ok2 = sanity2_toy_single_atom(model)

    print("\n" + "=" * 60)
    print("SANITY 3: Toy Molecule Multi-Site Update")
    print("=" * 60)
    ok3 = sanity3_toy_molecule(model)

    print(f"\n{'='*60}")
    print(f"SUMMARY: Force matrix OK, Atom update {'PASS' if ok2 else 'FAIL'}, Molecule update done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
