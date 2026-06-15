#!/usr/bin/env python3
"""DecompDiff + KAG — self-contained, no ESField imports to avoid conflicts."""
import sys, os, json, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")

os.chdir("/root/baselines/DecompDiff/code/DecompDiff-main")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/scripts")

import utils.transforms as trans
from models.decompdiff import DecompScorePosNet3D
from utils.data import ProteinLigandData, PDBProtein
from datasets.pl_data import torchify_dict
from scripts.sample_diffusion_decomp import sample_diffusion_ligand_decomp
from torch_geometric.transforms import Compose

# ═══════════════════════════════════════════════════════════════════════════
# Inline E_site + compatibility matrix (avoids ESField models/ namespace conflict)
# ═══════════════════════════════════════════════════════════════════════════

ATOM_TYPE_VOCAB = ("unknown","C_sp3","C_aromatic","N_donor","N_acceptor",
                    "O_acceptor","S","P","halogen","charged","B")
N_ATOM_TYPES = len(ATOM_TYPE_VOCAB)

HEW_ENV_ORDER = ["hydrophobic","polar_unsatisfied","mixed","buried"]

# Paper Table 10 compatibility matrix [4 envs, 11 types]
_COMPAT = torch.tensor([
    [-0.5, 1.0, 1.0, -0.8, -0.8, -0.8, 0.3, -0.5, 1.0, -1.0, 0.0],  # hydrophobic
    [-0.5, -0.8, -0.5, 1.0, 1.0, 1.0, -0.3, -0.3, -0.8, -0.5, 0.0], # polar_unsat
    [-0.5, 0.5, 0.5, 0.3, 0.3, 0.3, -0.3, -0.3, 0.5, -0.5, 0.0],    # mixed
    [-0.5, 0.8, 0.8, -0.3, -0.3, -0.3, 0.5, -0.3, 0.8, -0.8, 0.0],   # buried
])


def classify_hew(site):
    f = site.get("features", {})
    hb, hy, nd = f.get("hbond_count",0), f.get("hydrophobic_contact_count",0), f.get("nearest_protein_distance",4.0)
    if nd < 2.5: return 3  # buried
    if hy >= 4 and hb <= 1: return 0  # hydrophobic
    if hb <= 1 and hy <= 2: return 1  # polar_unsat
    return 2  # mixed


class SiteEnergy:
    """Minimal E_site — same formula as SiteCompatibilityEnergy."""
    def __init__(self, site_map, sigma=3.0, tau=10.0):
        hew = [s for s in site_map["sites"] if s.get("site_type") == "high_energy_water"]
        self.centers = torch.tensor([s["center"] for s in hew], dtype=torch.float32) if hew else torch.zeros(0,3)
        self.env_idx = torch.tensor([classify_hew(s) for s in hew], dtype=torch.long) if hew else torch.zeros(0,dtype=torch.long)
        self.confs = torch.tensor([s.get("confidence",1.0) for s in hew], dtype=torch.float32) if hew else torch.zeros(0)
        self.sigma = sigma; self.tau = tau; self.n = len(hew)
        self.compat = _COMPAT.clone()

    def to(self, d):
        self.centers = self.centers.to(d); self.env_idx = self.env_idx.to(d)
        self.confs = self.confs.to(d); self.compat = self.compat.to(d)
        return self

    def __call__(self, x, atom_probs=None):
        if self.n == 0: return x.new_zeros(())
        rel = x[:, None, :] - self.centers[None, :, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)
        gauss = torch.exp(-dist**2 / (2*self.sigma**2)) * self.confs[None,:]
        if atom_probs is not None:
            ntp = min(atom_probs.shape[-1], N_ATOM_TYPES)
            env_c = self.compat[self.env_idx, :ntp]
            compat = torch.matmul(atom_probs[:, :ntp], env_c.T)
        else:
            compat = torch.ones_like(gauss)
        pair = compat * gauss
        per_atom = pair.sum(dim=-1)
        return -(1.0/self.tau) * torch.logsumexp(self.tau * per_atom, dim=0)


class KAGDrift:
    def __init__(self, energy, lam=1.0, mode="full", anchors=None):
        self.e = energy; self.lam = lam; self.mode = mode; self.anchors = anchors or []
    def to(self, d): self.e.to(d); return self
    def compute_gradient(self, xt, batch_ligand):
        x = xt.detach().clone().requires_grad_(True)
        ap = torch.ones(x.shape[0], N_ATOM_TYPES, device=x.device) / N_ATOM_TYPES
        en = self.e(x, ap)
        g = torch.autograd.grad(en, x)[0]
        if self.mode == "com" and self.anchors:
            idx = torch.tensor(self.anchors, device=x.device, dtype=torch.long)
            cg = g[idx].mean(0)
            g = torch.zeros_like(g); g[idx] = cg.unsqueeze(0).expand(idx.numel(), 3)
        return -self.lam * g.detach(), float(en.detach().cpu())
    def opt(self): return {'type': 'kag_esite', 'drift_fn': self}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

CKPT = "/root/autodl-tmp/checkpoints/DecompDiff/uni_o2_bond.pt"
POCKET = "3mfw"; DEV = "cuda:0"; N = 10; STEPS = 200
PPDB = f"/root/autodl-tmp/data/PDB/P-L/2001-2010/{POCKET}/{POCKET}_protein.pdb"
SM = f"/root/ESField/experiments/targetdiff_replication/site_maps/{POCKET}_site_map.json"
OUT = f"/root/ESField/results/decompdiff_3mfw/{POCKET}"

print(f"Loading DecompDiff ({POCKET})...")
ck = torch.load(CKPT, map_location=DEV, weights_only=False); cfg = ck["config"]
pf = trans.FeaturizeProteinAtom()
lf = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
model = DecompScorePosNet3D(cfg.model, pf.protein_feature_dim, lf.ligand_feature_dim, num_classes=lf.ligand_feature_dim, prior_atom_types=lf.atom_types_prob, prior_bond_types=lf.bond_types_prob).to(DEV)
model.load_state_dict(ck["model"], strict=False); model.eval()
print(f"  {sum(p.numel() for p in model.parameters()):,} params")

prot = PDBProtein(PPDB)
pdict = torchify_dict(prot.to_dict_atom())
data = ProteinLigandData.from_protein_ligand_dicts(
    protein_dict=pdict, ligand_dict={
        'element': torch.empty([0], dtype=torch.long), 'pos': torch.empty([0,3]),
        'atom_feature': torch.empty([0,8]), 'bond_index': torch.empty([2,0]),
        'bond_type': torch.empty([0], dtype=torch.long),
    }).to(DEV)
print(f"  Pocket: {data.protein_pos.shape[0]} atoms")

sm = json.loads(open(SM).read())
se = SiteEnergy(sm).to(DEV)
print(f"  HEW sites: {se.n}")

with open("utils/evaluation/arm_num_config.pkl",'rb') as f: ac = pickle.load(f)
with open("utils/evaluation/scaffold_num_config.pkl",'rb') as f: sc = pickle.load(f)
tr = Compose([pf, lf])

for cond, drift in [
    ("unguided", None),
    ("full_gradient", [KAGDrift(se, lam=5.0, mode="full").to(DEV).opt()]),
    ("kag", [KAGDrift(se, lam=10.0, mode="com", anchors=[0,1]).to(DEV).opt()]),
]:
    od = f"{OUT}/{cond}"; os.makedirs(od, exist_ok=True)
    sp = f"{od}/molecules.sdf"
    if os.path.exists(sp) and os.path.getsize(sp) > 1000:
        print(f"  {cond}: skip"); continue
    print(f"  {cond} ({N} mols, {STEPS} steps)...", end=" ", flush=True)
    t0 = time.time()
    res = sample_diffusion_ligand_decomp(
        model=model, data=data, init_transform=tr,
        num_samples=N, batch_size=min(N,5), device=DEV, prior_mode='subpocket',
        num_steps=STEPS, center_pos_mode='none', num_atoms_mode='prior',
        arms_natoms_config=ac, scaffold_natoms_config=sc,
        atom_enc_mode='add_aromatic', bond_fc_mode='fc', energy_drift_opt=drift,
    )
    t = time.time()-t0
    mols = []
    for r in res:
        try:
            ppos, pv = r['pred_pos'], r['pred_v']
            if pv.ndim > 1: pv = np.argmax(pv, axis=1)
            import utils.reconstruct as rcn
            mol = rcn.reconstruct_from_generated(ppos, pv, None)
            if mol:
                try: Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except: pass
                mols.append(mol)
        except: pass
    w = Chem.SDWriter(sp); w.SetKekulize(False); [w.write(m) for m in mols]; w.close()
    print(f"{len(mols)} mols, {t:.0f}s")
    json.dump({"cond":cond,"n":len(mols),"time":t}, open(f"{od}/meta.json","w"))

print("\n✓ DecompDiff experiment complete")
