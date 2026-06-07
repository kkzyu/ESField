#!/usr/bin/env python3
"""Pocket-level validation: split by protein_id, not random pair."""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.potential_network import CompatibilityPotentialV5, PotentialConfig

# Load v5
ckpt = torch.load(ROOT / 'experiments/potential_training/v5/potential_v5_epoch_0030.pt', map_location='cpu')
cfg = PotentialConfig(**{k: ckpt['config'][k] for k in ['atom_embed_dim','site_embed_dim','hidden_dim','num_layers']})
model = CompatibilityPotentialV5(cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load pairs
pairs = []
with open(ROOT / 'experiments/pdbbind_water_sites/v5_pairs/train_pairs.jsonl') as f:
    for line in f:
        pairs.append(json.loads(line))

by_protein = defaultdict(list)
for p in pairs:
    by_protein[p['protein_id']].append(p)
protein_ids = sorted(by_protein.keys())
print(f"Total pairs: {len(pairs)}, pockets: {len(protein_ids)}")

np.random.seed(42)
train_prots = set(np.random.choice(protein_ids, int(len(protein_ids)*0.8), replace=False))
valid_prots = set(protein_ids) - train_prots
valid_pairs = [p for p in pairs if p['protein_id'] in valid_prots]
print(f"Valid: {len(valid_pairs)} pairs ({len(valid_prots)} pockets)")

ATM = {'unknown':0,'C_sp3':1,'C_aromatic':2,'N_donor':3,'N_acceptor':4,'O_acceptor':5,'S':6,'halogen':7,'P':8}
STM = {'unknown':0,'high_energy_water':1,'stable_water':2,'hydrophobic_cavity':3}

def score(batch):
    at = torch.tensor([ATM.get(p['atom_type'],0) for p in batch], dtype=torch.long)
    st = torch.tensor([STM.get(p['site_type'],0) for p in batch], dtype=torch.long)
    rl = torch.tensor([[p.get('rel_x',0),p.get('rel_y',0),p.get('rel_z',0)] for p in batch], dtype=torch.float32)
    dd = torch.tensor([p['distance'] for p in batch], dtype=torch.float32)
    rd = torch.tensor([p.get('site_radius',1.4) for p in batch], dtype=torch.float32)
    cf = torch.tensor([p.get('site_confidence',1.0) for p in batch], dtype=torch.float32)
    with torch.no_grad():
        return (-model(at, st, rl, dd, rd, cf)).numpy()

# Ordinary AUC
vs, vl = [], []
for i in range(0, len(valid_pairs), 512):
    b = valid_pairs[i:i+512]
    vs.extend(score(b).tolist())
    vl.extend([p['label'] for p in b])
auc_ord = roc_auc_score(vl, vs)
print(f"Pocket-level ordinary AUC: {auc_ord:.4f}")

# Distance-matched AUC
pos = [p for p in valid_pairs if p['label']==1]
neg = [p for p in valid_pairs if p['label']==0]
bins = [(1.5,2.5),(2.5,3.5),(3.5,5.0)]
ams, aml = [], []
for lo, hi in bins:
    bp = [p for p in pos if lo <= p['distance'] < hi]
    bn = [p for p in neg if lo <= p['distance'] < hi]
    n = min(len(bp), len(bn), 80)
    if n < 5: continue
    sp = np.random.choice(bp, n, replace=False)
    sn = np.random.choice(bn, n, replace=False)
    s = score(list(sp)+list(sn))
    ams.extend(s.tolist()); aml.extend([1]*n+[0]*n)
    print(f"  d in [{lo},{hi}): n={n}, AUC={roc_auc_score([1]*n+[0]*n, s):.4f}")

auc_m = roc_auc_score(aml, ams)
print(f"Pocket-level distance-matched AUC: {auc_m:.4f}")
print(f"Random-split dist-matched: 0.989  →  Drop: {0.989-auc_m:.4f}")
print(f"{'RELIABLE' if auc_m>=0.80 else 'MARGINAL' if auc_m>=0.75 else 'NOT RELIABLE'}")

# --- Type-shuffled AUC ---
print("\n=== Type-Shuffled AUC ===")
np.random.seed(42)
shuf_s, shuf_l = [], []
ATM2 = {'unknown':0,'C_sp3':1,'C_aromatic':2,'N_donor':3,'N_acceptor':4,'O_acceptor':5,'S':6,'halogen':7,'P':8}
for i in range(0, len(pairs), 512):
    b = pairs[i:i+512]
    at = torch.tensor([ATM2.get(p['atom_type'],0) for p in b], dtype=torch.long)
    st_arr = np.array([STM.get(p['site_type'],0) for p in b])
    np.random.shuffle(st_arr)
    st = torch.tensor(st_arr, dtype=torch.long)
    rl = torch.tensor([[p.get('rel_x',0),p.get('rel_y',0),p.get('rel_z',0)] for p in b], dtype=torch.float32)
    dd = torch.tensor([p['distance'] for p in b], dtype=torch.float32)
    rd = torch.tensor([p.get('site_radius',1.4) for p in b], dtype=torch.float32)
    cf = torch.tensor([p.get('site_confidence',1.0) for p in b], dtype=torch.float32)
    with torch.no_grad():
        shuf_s.extend((-model(at, st, rl, dd, rd, cf)).numpy().tolist())
    shuf_l.extend([p['label'] for p in b])

auc_shuf = roc_auc_score(shuf_l, shuf_s)
print(f"Ordinary AUC:       {auc_ord:.4f}")
print(f"Type-shuffled AUC:  {auc_shuf:.4f}")
print(f"Shuffle drop:       {auc_ord - auc_shuf:+.4f}")
print(f"{'STRONG type signal' if auc_ord-auc_shuf > 0.10 else 'MODERATE' if auc_ord-auc_shuf > 0.05 else 'WEAK'}")

# --- Per-site-type alpha analysis ---
print("\n=== Per-Site-Type Alpha (compatibility coefficient) ===")
alpha_by_label_site = {0: {}, 1: {}}
for i in range(0, min(len(pairs), 1000), 128):
    b = pairs[i:i+128]
    at = torch.tensor([ATM2.get(p['atom_type'],0) for p in b], dtype=torch.long)
    st = torch.tensor([STM.get(p['site_type'],0) for p in b], dtype=torch.long)
    rl = torch.tensor([[p.get('rel_x',0),p.get('rel_y',0),p.get('rel_z',0)] for p in b], dtype=torch.float32)
    dd = torch.tensor([p['distance'] for p in b], dtype=torch.float32)
    rd = torch.tensor([p.get('site_radius',1.4) for p in b], dtype=torch.float32)
    cf = torch.tensor([p.get('site_confidence',1.0) for p in b], dtype=torch.float32)
    with torch.no_grad():
        a, _ = model.get_coefficients(at, st, rl, dd, rd, cf)
    for p, av in zip(b, a.tolist()):
        st = p['site_type']
        lb = p['label']
        alpha_by_label_site[lb].setdefault(st, []).append(av)

for st_name in ['high_energy_water', 'stable_water', 'hydrophobic_cavity']:
    pos_a = np.mean(alpha_by_label_site[1].get(st_name, [0]))
    neg_a = np.mean(alpha_by_label_site[0].get(st_name, [0]))
    sig = 'POS>>NEG' if pos_a > 2*neg_a else ('POS>NEG' if pos_a > neg_a else 'NO DIFF')
    print(f"  {st_name:<22} pos_alpha={pos_a:.4f}  neg_alpha={neg_a:.4f}  [{sig}]")
