#!/usr/bin/env python3
"""DecompDiff + KAG — 3mfw validation (Exp 3.4).

Run from DecompDiff directory with ESField src in PYTHONPATH.
Handles import shadowing between ESField src/utils and DecompDiff utils.
"""

import sys, os, json, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ── PATH SETUP — critical ordering ──
# Remove ESField src to avoid utils shadowing
DECOMP_ROOT = "/root/baselines/DecompDiff/code/DecompDiff-main"
ESFIELD_SRC = "/root/ESField/src"
sys.path = [p for p in sys.path if 'ESField/src' not in p]
sys.path.insert(0, DECOMP_ROOT)
sys.path.insert(0, DECOMP_ROOT + "/scripts")
os.chdir(DECOMP_ROOT)

# DecompDiff imports
import utils.transforms as trans
from models.decompdiff import DecompScorePosNet3D
from utils.data import ProteinLigandData, PDBProtein
from utils.evaluation import atom_num
from datasets.pl_data import torchify_dict
from scripts.sample_diffusion_decomp import sample_diffusion_ligand_decomp
from torch_geometric.transforms import Compose

# ESField imports (after DecompDiff paths)
sys.path.insert(0, ESFIELD_SRC)
from guidance.latent_guidance import (
    SiteCompatibilityEnergy, build_site_energy_from_map, N_ATOM_TYPES,
)

CKPT = "/root/autodl-tmp/checkpoints/DecompDiff/uni_o2_bond.pt"
POCKET = "3mfw"
DEVICE = "cuda:0"
N_MOLS = 10  # quick test
NUM_STEPS = 200  # reduced for faster testing (full: 1000)

PPDB = f"/root/autodl-tmp/data/PDB/P-L/2001-2010/{POCKET}/{POCKET}_protein.pdb"
SITEMAP = f"/root/ESField/experiments/targetdiff_replication/site_maps/{POCKET}_site_map.json"
OUT_DIR = f"/root/ESField/results/decompdiff_3mfw/{POCKET}"


# ═══════════════════════════════════════════════════════════════════════════
# KAG Energy Drift
# ═══════════════════════════════════════════════════════════════════════════

class KAGEnergyDrift:
    def __init__(self, site_energy, lambda_guide=1.0, mode="full_gradient",
                 anchor_indices=None):
        self.site_energy = site_energy
        self.lambda_guide = lambda_guide
        self.mode = mode
        self.anchor_indices = anchor_indices or []

    def to(self, d): self.site_energy.to(d); return self

    def compute_gradient(self, xt, batch_ligand, atom_type_probs=None):
        device = xt.device
        xt_in = xt.detach().clone().requires_grad_(True)
        if atom_type_probs is None:
            atom_type_probs = torch.ones(xt.shape[0], N_ATOM_TYPES, device=device) / N_ATOM_TYPES
        energy = self.site_energy(xt_in, atom_type_probs=atom_type_probs)
        grad = torch.autograd.grad(energy, xt_in)[0]

        if self.mode == "com_projection" and len(self.anchor_indices) > 0:
            idx = torch.tensor(self.anchor_indices, device=device, dtype=torch.long)
            if idx.numel() > 0:
                com_grad = grad[idx].mean(dim=0)
                grad = torch.zeros_like(grad)
                grad[idx] = com_grad.unsqueeze(0).expand(idx.numel(), 3)

        return -self.lambda_guide * grad.detach(), float(energy.detach().cpu())

    def get_drift_opt(self):
        return {'type': 'kag_esite', 'drift_fn': self}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

print("Loading DecompDiff model...")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
config = ckpt["config"]
pf = trans.FeaturizeProteinAtom()
lf = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
model = DecompScorePosNet3D(config.model, pf.protein_feature_dim, lf.ligand_feature_dim).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

# Prepare pocket
protein = PDBProtein(PPDB)
pd = torchify_dict(protein.to_dict_atom())
pocket_data = ProteinLigandData.from_protein_ligand_dicts(
    protein_dict=pd, ligand_dict={
        'element': torch.empty([0], dtype=torch.long),
        'pos': torch.empty([0, 3]), 'atom_feature': torch.empty([0, 8]),
        'bond_index': torch.empty([2, 0]), 'bond_type': torch.empty([0], dtype=torch.long),
    }).to(DEVICE)
print(f"  Pocket: {pocket_data.protein_pos.shape[0]} atoms")

# Build site energy
sm = json.loads(open(SITEMAP).read())
se = build_site_energy_from_map(sm, sigma_distance=3.0).to(DEVICE)
print(f"  HEW sites: {se.n_sites}")

# Prior configs
with open("utils/evaluation/arm_num_config.pkl", 'rb') as f: arms_cfg = pickle.load(f)
with open("utils/evaluation/scaffold_num_config.pkl", 'rb') as f: scaff_cfg = pickle.load(f)

transform = Compose([pf, lf])

for cond, drift_opt in [
    ("unguided", None),
    ("full_gradient", [KAGEnergyDrift(se, lambda_guide=5.0, mode="full_gradient").to(DEVICE).get_drift_opt()]),
    ("kag", [KAGEnergyDrift(se, lambda_guide=10.0, mode="com_projection", anchor_indices=[0,1]).to(DEVICE).get_drift_opt()]),
]:
    out_dir = f"{OUT_DIR}/{cond}"
    os.makedirs(out_dir, exist_ok=True)
    sdf_path = f"{out_dir}/molecules.sdf"
    if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 1000:
        print(f"  {cond}: exists, skip")
        continue

    print(f"  {cond} ({N_MOLS} mols, {NUM_STEPS} steps)...", end=" ", flush=True)
    t0 = time.time()
    result = sample_diffusion_ligand_decomp(
        model=model, data=pocket_data, init_transform=transform,
        num_samples=N_MOLS, batch_size=min(N_MOLS, 5), device=DEVICE,
        prior_mode='subpocket', num_steps=NUM_STEPS, center_pos_mode='none',
        num_atoms_mode='prior', arms_natoms_config=arms_cfg,
        scaffold_natoms_config=scaff_cfg, atom_enc_mode='add_aromatic',
        bond_fc_mode='fc', energy_drift_opt=drift_opt,
    )
    elapsed = time.time() - t0

    mols = []
    for r in result:
        try:
            pred_pos = r['pred_pos']
            pred_v = r['pred_v']
            if pred_v.ndim > 1: pred_v = np.argmax(pred_v, axis=1)
            mol = __import__('utils').reconstruct.reconstruct_from_generated(pred_pos, pred_v, None)
            if mol is not None:
                try: Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except: pass
                mols.append(mol)
        except: pass

    w = Chem.SDWriter(sdf_path); w.SetKekulize(False)
    [w.write(m) for m in mols if m is not None]
    w.close()

    # Basic metrics
    if mols:
        from rdkit.Chem import QED
        qeds = [QED.qed(m) for m in mols if m is not None]
        print(f"{len(mols)} mols, {elapsed:.0f}s, QED={np.mean(qeds):.3f}±{np.std(qeds):.3f}")
    else:
        print(f"0 mols, {elapsed:.0f}s")

    json.dump({"condition": cond, "n": len(mols), "time_s": elapsed},
              open(f"{out_dir}/metadata.json", "w"), indent=2)

print("\n✓ DecompDiff 3mfw experiment complete")
