#!/usr/bin/env python3
"""Aggregation sweep: compare molecule-level energy functions on baseline mols."""
import json, torch, numpy as np, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
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

def pair_energies(mol, sm):
    atoms = _extract_atoms_from_mol(mol)
    hew = [s for s in sm["sites"] if s["site_type"]=="high_energy_water"]
    pairs = []
    for site in hew:
        sc = site["center"]
        for a in atoms:
            if is_compatible_atom_site(a["atom_type"], a["atomic_number"], "high_energy_water"):
                d = calc_dist(a["coord"], sc)
                at_t = torch.tensor([ATM.get(a["atom_type"], 0)])
                st_t = torch.tensor([1])
                rel_t = torch.tensor([[a["coord"][0]-sc[0], a["coord"][1]-sc[1], a["coord"][2]-sc[2]]], dtype=torch.float32)
                dist_t = torch.tensor([d], dtype=torch.float32)
                rad_t = torch.tensor([site.get("radius", 1.4)], dtype=torch.float32)
                conf_t = torch.tensor([site.get("confidence", 1.0)], dtype=torch.float32)
                with torch.no_grad():
                    e = model(at_t, st_t, rel_t, dist_t, rad_t, conf_t).item()
                pairs.append((e, site["site_id"]))
    return pairs

def agg_mol(pairs_by_site, method, tau=1.0):
    vals = []
    for sid, pairs in pairs_by_site.items():
        if not pairs: continue
        e_list = [p[0] for p in pairs]
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
    return np.mean(vals) if method != "best_site_min" else vals[0]

def compute_rho(mols, sm, method, tau=1.0):
    energies, wells, natoms = [], [], []
    for m in mols:
        try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except: continue
        pairs = pair_energies(m, sm)
        by_site = {}
        for e, sid in pairs: by_site.setdefault(sid, []).append((e, sid))
        e_mol = agg_mol(by_site, method, tau)
        energies.append(e_mol)
        wells.append(hew_well(m, sm))
        natoms.append(len(_extract_atoms_from_mol(m)))
    if len(energies) < 5: return float('nan'), float('nan')
    if method == "sum_norm": energies = [e/n for e, n in zip(energies, natoms)]
    from scipy.stats import spearmanr
    r, p = spearmanr(energies, wells)
    return r, p

methods = [
    ("sum", None), ("sum_norm", None),
    ("per_site_min", None), ("per_site_top1", None), ("per_site_top2", None),
    ("per_site_softmin_t0.5", 0.5), ("per_site_softmin_t1.0", 1.0), ("per_site_softmin_t2.0", 2.0),
    ("best_site_min", None),
]

all_data = {}
for pid in POCKETS:
    sm = json.load(open(SM / "correct" / f"{pid}_site_map.json"))
    mols = [m for m in Chem.SDMolSupplier(str(OUT / f"{pid}_baseline" / "molecules.sdf"), sanitize=False) if m is not None]
    all_data[pid] = (mols, sm)

print("=" * 110)
print("AGGREGATION SWEEP: Spearman ρ(E_mol, |d-3.0|)")
print("=" * 110)
print(f"  {'Method':<28}", end="")
for pid in POCKETS: print(f"  {pid:>8}", end="")
print(f"  {'Mean ρ':>8}  {'≥0.5':>5}")
print(f"  {'-'*110}")

best_method = None; best_mean_rho = -99
for method, tau in methods:
    print(f"  {method:<28}", end="")
    rhos = []
    for pid in POCKETS:
        r, p = compute_rho(all_data[pid][0], all_data[pid][1], method, tau if tau else 1.0)
        rhos.append(r)
        mk = "*" if r > 0.5 else ("!" if r < 0 else "")
        print(f"  {r:>+7.3f}{mk}", end="")
    mean_r = np.nanmean(rhos)
    n_good = sum(1 for r in rhos if not np.isnan(r) and r > 0.5)
    print(f"  {mean_r:>+8.3f}  {n_good:>5}/5")
    if mean_r > best_mean_rho: best_mean_rho = mean_r; best_method = method

print(f"\n  Best: {best_method} (mean ρ = {best_mean_rho:+.3f})")

# Per-pocket best
print(f"\n{'='*110}")
print("PER-POCKET BEST:")
print(f"  {'Pocket':<8}", end="")
for method, _ in methods: print(f"  {method[:15]:>15}", end="")
print(f"  {'Best':>20}")
print(f"  {'-'*160}")
for pid in POCKETS:
    mols, sm = all_data[pid]
    print(f"  {pid:<8}", end="")
    best_r = -99; best_m = ""
    for method, tau in methods:
        r, p = compute_rho(mols, sm, method, tau if tau else 1.0)
        print(f"  {r:>+15.3f}", end="")
        if r > best_r: best_r = r; best_m = method
    print(f"  {best_m:>20} ({best_r:+.3f})")

print(f"\n  * = ρ>0.5  ! = ρ<0")
