#!/usr/bin/env python3
"""Train CompatibilityPotentialV5 — HEW-focused with hand-crafted energy shape.

Key differences from v4 training:
  - Uses CompatibilityPotentialV5 (hand-crafted E(d) × learned α,β)
  - HEW-focused training data with distance-matched negatives
  - Validation: distance-matched AUC, force matrix, energy curves
"""

import json, sys, time, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.potential_network import CompatibilityPotentialV5, PotentialConfig
from models.atom_features import atom_type_to_index
from models.site_features import site_type_to_index

ATOM_TYPE_MAP = {'unknown': 0, 'C_sp3': 1, 'C_aromatic': 2, 'N_donor': 3,
                 'N_acceptor': 4, 'O_acceptor': 5, 'S': 6, 'halogen': 7, 'P': 8}
SITE_TYPE_MAP = {'unknown': 0, 'high_energy_water': 1, 'stable_water': 2, 'hydrophobic_cavity': 3}


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def make_batch(pairs, device):
    at = torch.tensor([ATOM_TYPE_MAP.get(p['atom_type'], 0) for p in pairs], dtype=torch.long, device=device)
    st = torch.tensor([SITE_TYPE_MAP.get(p['site_type'], 0) for p in pairs], dtype=torch.long, device=device)
    rel = torch.tensor([[p.get('rel_x', 0), p.get('rel_y', 0), p.get('rel_z', 0)] for p in pairs],
                       dtype=torch.float32, device=device)
    dist = torch.tensor([p['distance'] for p in pairs], dtype=torch.float32, device=device)
    rad = torch.tensor([p.get('site_radius', 1.4) for p in pairs], dtype=torch.float32, device=device)
    conf = torch.tensor([p.get('site_confidence', 1.0) for p in pairs], dtype=torch.float32, device=device)
    labels = torch.tensor([p['label'] for p in pairs], dtype=torch.float32, device=device)
    return at, st, rel, dist, rad, conf, labels


def margin_loss(energies, labels, pos_target=-1.0, neg_target=1.0):
    """Margin loss: push positives below pos_target, negatives above neg_target."""
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    loss = torch.tensor(0.0, device=energies.device)
    if pos_mask.any():
        loss = loss + torch.relu(energies[pos_mask] - pos_target).mean()
    if neg_mask.any():
        loss = loss + torch.relu(neg_target - energies[neg_mask]).mean()
    return loss


def compute_auc(model, pairs, device, batch_size=2048):
    """Compute ordinary AUC on a set of pairs."""
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            at, st, rel, dist, rad, conf, labels = make_batch(batch, device)
            energies = model(at, st, rel, dist, rad, conf)
            all_scores.extend((-energies).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(all_labels, all_scores)


def compute_distance_matched_auc(model, pairs, device):
    """Compute AUC within each distance bin, and pooled."""
    from sklearn.metrics import roc_auc_score
    pos = [p for p in pairs if p['label'] == 1]
    neg = [p for p in pairs if p['label'] == 0]
    bins = [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 8.0)]
    model.eval()
    all_scores, all_labels = [], []
    bin_aucs = {}

    for lo, hi in bins:
        bin_pos = [p for p in pos if lo <= p['distance'] < hi]
        bin_neg = [p for p in neg if lo <= p['distance'] < hi]
        n = min(len(bin_pos), len(bin_neg))
        if n < 5:
            continue
        np.random.seed(42)
        sp = np.random.choice(bin_pos, n, replace=False)
        sn = np.random.choice(bin_neg, n, replace=False)
        batch = list(sp) + list(sn)
        with torch.no_grad():
            at, st, rel, dist, rad, conf, _ = make_batch(batch, device)
            energies = model(at, st, rel, dist, rad, conf)
        scores = (-energies).cpu().numpy().tolist()
        labels = [1]*n + [0]*n
        all_scores.extend(scores)
        all_labels.extend(labels)
        bin_aucs[(lo, hi)] = roc_auc_score(labels, scores)

    pooled_auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
    return pooled_auc, bin_aucs


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    train_pairs = load_pairs(args.train_pairs)
    valid_pairs = load_pairs(args.valid_pairs)
    print(f"Train: {len(train_pairs)} pairs, Valid: {len(valid_pairs)} pairs")

    # Model
    cfg = PotentialConfig(
        atom_embed_dim=args.atom_embed_dim,
        site_embed_dim=args.site_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rbf_bins=args.rbf_bins,
        cutoff=args.cutoff,
        energy_clip=args.energy_clip,
    )
    model = CompatibilityPotentialV5(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train_pairs)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[i:i+args.batch_size]
            at, st, rel, dist, rad, conf, labels = make_batch(batch, device)
            energies = model(at, st, rel, dist, rad, conf)
            loss = margin_loss(energies, labels, args.pos_target, args.neg_target)
            # Add weak regularization: encourage alpha>0 for positives, beta>0 for negatives
            alpha, beta = model.get_coefficients(at, st, rel, dist, rad, conf)
            pos_mask = labels > 0.5
            if pos_mask.any():
                loss = loss + 0.01 * torch.relu(0.5 - alpha[pos_mask]).mean()
            if (~pos_mask).any():
                loss = loss + 0.01 * torch.relu(0.5 - beta[~pos_mask]).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            auc_ord = compute_auc(model, valid_pairs, device)
            auc_matched, bin_aucs = compute_distance_matched_auc(model, valid_pairs, device)
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}  AUC_ord={auc_ord:.4f}  AUC_matched={auc_matched:.4f}  ", end="")
            for (lo, hi), auc in sorted(bin_aucs.items()):
                print(f"[{lo:.1f}-{hi:.1f}]:{auc:.3f} ", end="")
            print()

            if auc_matched > best_auc:
                best_auc = auc_matched
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "atom_embed_dim": args.atom_embed_dim,
                        "site_embed_dim": args.site_embed_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                    },
                    "auc_ordinary": auc_ord,
                    "auc_distance_matched": auc_matched,
                    "bin_aucs": {f"{lo}-{hi}": auc for (lo, hi), auc in bin_aucs.items()},
                }
                torch.save(ckpt, out_dir / f"potential_v5_epoch_{epoch:04d}.pt")
                print(f"  => saved (best distance-matched AUC: {best_auc:.4f})")

    print(f"\nBest distance-matched AUC: {best_auc:.4f}")
    return best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", default=f"{ROOT}/experiments/pdbbind_water_sites/v5_pairs/train_pairs.jsonl")
    parser.add_argument("--valid-pairs", default=f"{ROOT}/experiments/pdbbind_water_sites/v5_pairs/valid_pairs.jsonl")
    parser.add_argument("--output-dir", default=f"{ROOT}/experiments/potential_training/v5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--atom-embed-dim", type=int, default=32)
    parser.add_argument("--site-embed-dim", type=int, default=32)
    parser.add_argument("--rbf-bins", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--energy-clip", type=float, default=5.0)
    parser.add_argument("--pos-target", type=float, default=-0.5)
    parser.add_argument("--neg-target", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
