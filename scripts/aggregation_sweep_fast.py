#!/usr/bin/env python3
"""Aggregation sweep — optimized with batched forward pass."""
import json, torch, numpy as np, sys
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.posu import _extract_atoms_from_mol
from utils.chemistry import is_compatible_atom_site
from utils.geometry import distance as calc_dist
from models.potential_network import CompatibilityPotentialV5, PotentialConfig

OUT = ROOT / "experiments/pdbbind_water_sites/v5_mechanism_test"
SM = ROOT / "experiments/pdbbind_water_sites/test_sites"
D0 = 3.0
POCKETS = ["3ohi", "2clh", "3mfw", "4bis", "1sle"]
ATM = {"unknown":0,"C_sp3":1,"C_aromatic":2,"N_donor":3,"N_acceptor":4,"O_acceptor":5,"S":6,"halogen":7,"P":8}

ckpt = torch.load(ROOT / "experiments/potential_training/v5/potential_v5_epoch_0030.pt", map_location="cpu")
cfg = PotentialConfig(**{k: ckpt["config"][k] for k in ["atom_embed_dim","site_embed_dim","hidden_dim","num_layers"]})
model = CompatibilityPotentialV5(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def hew_well(mol, sm):
    atoms = _extract_atoms_from_mol(mol)
    hew = [s for s in sm["sites"] if s["site_type"]=="high_energy_water"]
    best = 10.0
    for site in hew:
        for a in atoms:
            if is_compatible_atom_site(a["atom_type"], a["atomic_number"], "high_energy_water"):
                d = calc_dist(a["coord"], tuple(site["center"]))
                if abs(d - D0) < best: best = abs(d - D0)
    return best

def all_mol_energies(mols, sm):
    """Return list of (per-mol dict of {site_id: [energies]}) for all mols."""
    results = []
    # Collect ALL pairs across all molecules
    all_at, all_st, all_rel, all_dist, all_rad, all_conf = [], [], [], [], [], []
    mol_map = []  # (mol_idx, site_id) for each pair
    mol_atom_counts = []

    for mi, m in enumerate(mols):
        try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except:
            results.append({})
            mol_atom_counts.append(0)
            continue
        atoms = _extract_atoms_from_mol(m)
        mol_atom_counts.append(len(atoms))
        hew = [s for s in sm["sites"] if s["site_type"]=="high_energy_water"]
        by_site = {}
        for site in hew:
            sc = site["center"]
            for a in atoms:
                if is_compatible_atom_site(a["atom_type"], a["atomic_number"], "high_energy_water"):
                    d = calc_dist(a["coord"], sc)
                    all_at.append(ATM.get(a["atom_type"], 0))
                    all_st.append(1)  # HEW
                    all_rel.append([a["coord"][0]-sc[0], a["coord"][1]-sc[1], a["coord"][2]-sc[2]])
                    all_dist.append(d)
                    all_rad.append(site.get("radius", 1.4))
                    all_conf.append(site.get("confidence", 1.0))
                    mol_map.append((mi, site["site_id"]))
        results.append({})

    if not all_at: return results, mol_atom_counts

    # Batch forward pass
    at_t = torch.tensor(all_at, dtype=torch.long)
    st_t = torch.tensor(all_st, dtype=torch.long)
    rel_t = torch.tensor(all_rel, dtype=torch.float32)
    dist_t = torch.tensor(all_dist, dtype=torch.float32)
    rad_t = torch.tensor(all_rad, dtype=torch.float32)
    conf_t = torch.tensor(all_conf, dtype=torch.float32)
    with torch.no_grad():
        energies = model(at_t, st_t, rel_t, dist_t, rad_t, conf_t).tolist()

    for (mi, sid), e in zip(mol_map, energies):
        results[mi].setdefault(sid, []).append(e)

    return results, mol_atom_counts

def agg_score(pairs_by_site, method, tau=1.0):
    if not pairs_by_site: return 99.0
    vals = []
    for sid, e_list in pairs_by_site.items():
        if not e_list: continue
        if method == "sum": vals.extend(e_list)
        elif method == "per_site_min": vals.append(min(e_list))
        elif method == "per_site_top1": vals.append(sorted(e_list)[0])
        elif method == "per_site_top2": vals.append(np.mean(sorted(e_list)[:2]))
        elif method == "per_site_softmin":
            et = torch.tensor(e_list)
            w = torch.softmax(-et / tau, dim=0)
            vals.append((et * w).sum().item())
        elif method == "best_site_min": return min(e_list)
    if not vals: return 99.0
    return np.mean(vals)

def compute_rho(mols, sm, energies_by_mol, natoms, method, tau=1.0):
    scores, wells = [], []
    for mi, eb in enumerate(energies_by_mol):
        if not eb: continue
        e_mol = agg_score(eb, method, tau)
        if method == "sum_norm": e_mol = e_mol / max(1, natoms[mi])
        scores.append(e_mol)
        wells.append(hew_well(mols[mi], sm))
    if len(scores) < 5: return float('nan'), float('nan')
    from scipy.stats import spearmanr
    r, p = spearmanr(scores, wells)
    return r, p

methods = [
    ("sum", None), ("sum_norm", None),
    ("per_site_min", None), ("per_site_top1", None), ("per_site_top2", None),
    ("per_site_softmin_t0.5", 0.5), ("per_site_softmin_t1.0", 1.0), ("per_site_softmin_t2.0", 2.0),
    ("best_site_min", None),
]

# Pre-compute all pair energies (batched, single forward pass per pocket)
print("Computing pair energies (batched)...")
all_data = {}
for pid in POCKETS:
    sm = json.load(open(SM / "correct" / f"{pid}_site_map.json"))
    mols = [m for m in Chem.SDMolSupplier(str(OUT / f"{pid}_baseline" / "molecules.sdf"), sanitize=False) if m is not None]
    eb, na = all_mol_energies(mols, sm)
    all_data[pid] = (mols, sm, eb, na)
    print(f"  {pid}: {len(eb)} mols, {sum(len(v) for v in eb)} pairs")

print(f"\n{'='*110}")
print("AGGREGATION SWEEP: Spearman ρ(E_mol, |d-3.0|)")
print(f"{'='*110}")
print(f"  {'Method':<28}", end="")
for pid in POCKETS: print(f"  {pid:>8}", end="")
print(f"  {'Mean ρ':>8}  {'≥0.5':>5}")
print(f"  {'-'*110}")

best_method = None; best_mean_rho = -99
for method, tau in methods:
    print(f"  {method:<28}", end="")
    rhos = []
    for pid in POCKETS:
        mols, sm, eb, na = all_data[pid]
        r, p = compute_rho(mols, sm, eb, na, method, tau if tau else 1.0)
        rhos.append(r)
        mk = "*" if r > 0.5 else ("!" if r < 0 else "")
        print(f"  {r:>+7.3f}{mk}", end="")
    mean_r = np.nanmean(rhos)
    n_good = sum(1 for r in rhos if not np.isnan(r) and r > 0.5)
    print(f"  {mean_r:>+8.3f}  {n_good:>5}/5")
    if mean_r > best_mean_rho: best_mean_rho = mean_r; best_method = method

print(f"\n  Best: {best_method} (mean ρ = {best_mean_rho:+.3f})")

# Per-pocket
print(f"\n{'='*110}")
print("PER-POCKET:")
for pid in POCKETS:
    mols, sm, eb, na = all_data[pid]
    best_r = -99; best_m = ""
    parts = []
    for method, tau in methods:
        r, p = compute_rho(mols, sm, eb, na, method, tau if tau else 1.0)
        parts.append(f"{r:+.3f}")
        if r > best_r: best_r = r; best_m = method
    print(f"  {pid}: {' | '.join(parts)}  → best: {best_m} (ρ={best_r:+.3f})")

print(f"\n  * = ρ>0.5  ! = ρ<0")
