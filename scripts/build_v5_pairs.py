#!/usr/bin/env python3
"""Build HEW-focused training pairs for Potential v5.

Key changes from v4:
  - Same-distance wrong-type negatives: force model to learn type, not distance
  - Same-type wrong-site negatives: prevent atom-type-only shortcuts
  - Hard negatives at 2.5-3.5Å (weakest AUC zone for v4)
  - Balanced positive/negative ratio within each distance bin

Output: experiments/pdbbind_water_sites/v5_pairs/train_pairs.jsonl
"""

import json, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ATOM_TYPE_MAP = {'unknown': 0, 'C_sp3': 1, 'C_aromatic': 2, 'N_donor': 3,
                 'N_acceptor': 4, 'O_acceptor': 5, 'S': 6, 'halogen': 7, 'P': 8}
SITE_TYPE_MAP = {'unknown': 0, 'high_energy_water': 1, 'stable_water': 2, 'hydrophobic_cavity': 3}

# HEW environment classification (matching posu.py v2 logic)
def classify_hew_env(features):
    if not features:
        return 'mixed'
    hbond = features.get('hbond_count', 0)
    hydrophobic = features.get('hydrophobic_contact_count', 0)
    nearest_dist = features.get('nearest_protein_distance', 4.0)
    if nearest_dist < 2.5:
        return 'buried'
    if hydrophobic >= 4 and hbond <= 1:
        return 'hydrophobic'
    if hbond <= 1 and hydrophobic <= 2:
        return 'polar_unsatisfied'
    return 'mixed'

# Compatibility rules for negative generation
HEW_COMPAT_ENV = {
    'hydrophobic':       {'C_sp3', 'C_aromatic', 'halogen', 'S'},
    'polar_unsatisfied': {'O_acceptor', 'N_donor', 'N_acceptor'},
    'mixed':             {'C_sp3', 'C_aromatic', 'halogen', 'S', 'O_acceptor', 'N_donor', 'N_acceptor'},
    'buried':            {'C_sp3', 'halogen'},
}
ALL_ATOM_TYPES = ['C_sp3', 'C_aromatic', 'N_donor', 'N_acceptor', 'O_acceptor', 'S', 'halogen', 'P']


def is_compat_hew(atom_type, site_features):
    env = classify_hew_env(site_features)
    return atom_type in HEW_COMPAT_ENV.get(env, set())


def build_v5_pairs():
    random.seed(20260519)
    np.random.seed(20260519)

    # Load existing pairs
    existing = []
    with open(ROOT / "experiments/pdbbind_water_sites/train_pairs.jsonl") as f:
        for line in f:
            existing.append(json.loads(line))

    pos_all = [p for p in existing if p['label'] == 1]
    neg_all = [p for p in existing if p['label'] == 0]
    print(f"Loaded: {len(pos_all)} positive, {len(neg_all)} negative")

    # Filter for HEW positives only
    pos_hew = [p for p in pos_all if p['site_type'] == 'high_energy_water']
    print(f"HEW positives: {len(pos_hew)}")

    # Group HEW positives by distance bin
    dist_bins = [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 8.0)]
    pos_by_bin = defaultdict(list)
    for p in pos_hew:
        d = p['distance']
        for lo, hi in dist_bins:
            if lo <= d < hi:
                pos_by_bin[(lo, hi)].append(p)
                break

    print("HEW positives per distance bin:")
    for (lo, hi), items in sorted(pos_by_bin.items()):
        print(f"  [{lo:.1f}, {hi:.1f}): {len(items)}")

    # Build negatives
    v5_pairs = []
    v5_pairs.extend(pos_hew)  # Keep all HEW positives

    neg_types = {
        'same_distance_wrong_type': 0,
        'same_type_wrong_site': 0,
        'shuffled_site_type': 0,
        'original_negatives': 0,
    }

    # --- Negative type 1: same distance, wrong atom type ---
    for p in pos_hew:
        d = p['distance']
        features = p.get('site_features', {})
        env = classify_hew_env(features)
        compat_types = HEW_COMPAT_ENV.get(env, set())
        # Pick an incompatible atom type
        wrong_types = [t for t in ALL_ATOM_TYPES if t not in compat_types]
        if wrong_types:
            wrong_at = random.choice(wrong_types)
            neg_p = dict(p)
            neg_p['label'] = 0
            neg_p['atom_type'] = wrong_at
            neg_p['atom_type_idx'] = ATOM_TYPE_MAP.get(wrong_at, 0)
            neg_p['negative_type'] = 'same_distance_wrong_type'
            neg_p['label_strength'] = 1.0
            v5_pairs.append(neg_p)
            neg_types['same_distance_wrong_type'] += 1

    # --- Negative type 2: same atom type, wrong site type (HC or SW) ---
    pos_by_atom = defaultdict(list)
    for p in pos_hew:
        pos_by_atom[p['atom_type']].append(p)

    for at_type, items in pos_by_atom.items():
        sample_n = min(len(items), 200)
        sampled = random.sample(items, sample_n)
        for p in sampled:
            # Replace site type with something incompatible for this atom
            if at_type in {'C_sp3', 'C_aromatic', 'halogen', 'S'}:
                wrong_st = 'stable_water'  # hydrophobic atoms should not attract to SW
            else:
                wrong_st = 'hydrophobic_cavity'  # polar atoms should not attract to HC
            neg_p = dict(p)
            neg_p['label'] = 0
            neg_p['site_type'] = wrong_st
            neg_p['site_type_idx'] = SITE_TYPE_MAP.get(wrong_st, 0)
            neg_p['negative_type'] = 'same_type_wrong_site'
            neg_p['label_strength'] = 1.0
            v5_pairs.append(neg_p)
            neg_types['same_type_wrong_site'] += 1

    # --- Negative type 3: shuffled site type (keeps distance and atom) ---
    hew_site_types = [p['site_type'] for p in pos_hew]
    random.shuffle(hew_site_types)
    for i, p in enumerate(pos_hew):
        if i >= len(hew_site_types):
            break
        if hew_site_types[i] == p['site_type']:
            continue  # skip if same type after shuffle
        neg_p = dict(p)
        neg_p['label'] = 0
        new_st = hew_site_types[i]
        neg_p['site_type'] = new_st
        neg_p['site_type_idx'] = SITE_TYPE_MAP.get(new_st, 0)
        neg_p['negative_type'] = 'shuffled_site_type'
        neg_p['label_strength'] = 0.5  # weaker label (type could be compatible by chance)
        v5_pairs.append(neg_p)
        neg_types['shuffled_site_type'] += 1

    # --- Negative type 4: include a subset of original negatives as distractors ---
    sample_n_orig = min(len(neg_all), len(pos_hew) * 2)
    sampled_neg = random.sample(neg_all, sample_n_orig)
    for p in sampled_neg:
        neg_p = dict(p)
        neg_p['negative_type'] = 'original'
        v5_pairs.append(neg_p)
        neg_types['original_negatives'] += 1

    random.shuffle(v5_pairs)

    # Statistics
    n_pos = sum(1 for p in v5_pairs if p['label'] == 1)
    n_neg = sum(1 for p in v5_pairs if p['label'] == 0)
    print(f"\nV5 training set: {len(v5_pairs)} total ({n_pos} pos, {n_neg} neg, ratio={n_neg/n_pos:.1f}:1)")
    print("Negative types:")
    for k, v in neg_types.items():
        print(f"  {k}: {v}")

    # Distance-matched check: within each bin, pos/neg balance
    print("\nDistance-bin balance (HEW positives + same-distance negatives):")
    for (lo, hi) in sorted(pos_by_bin.keys()):
        bin_pos = sum(1 for p in v5_pairs if p['label'] == 1 and lo <= p.get('distance', 0) < hi and p.get('site_type') == 'high_energy_water')
        bin_neg = sum(1 for p in v5_pairs if p['label'] == 0 and lo <= p.get('distance', 0) < hi and p.get('negative_type') == 'same_distance_wrong_type')
        print(f"  [{lo:.1f}, {hi:.1f}): {bin_pos} pos, {bin_neg} neg")

    # Write
    out_dir = ROOT / "experiments/pdbbind_water_sites/v5_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_pairs.jsonl"
    with open(out_path, "w") as f:
        for p in v5_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved: {out_path}")

    # Also split a small validation set
    random.shuffle(v5_pairs)
    split = int(len(v5_pairs) * 0.85)
    train = v5_pairs[:split]
    valid = v5_pairs[split:]
    with open(out_dir / "train_pairs.jsonl", "w") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")
    with open(out_dir / "valid_pairs.jsonl", "w") as f:
        for p in valid:
            f.write(json.dumps(p) + "\n")
    print(f"Train: {len(train)}, Valid: {len(valid)}")


if __name__ == "__main__":
    build_v5_pairs()
