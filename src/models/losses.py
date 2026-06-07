"""Losses and lightweight validation metrics for compatibility potentials."""

from __future__ import annotations

from models.distance_encoding import require_torch

torch = require_torch()


def margin_energy_loss(energy, label, *, label_strength=None, margin_pos: float = -1.0, margin_neg: float = 1.0):
    label_strength = torch.ones_like(label) if label_strength is None else label_strength
    pos_loss = torch.nn.functional.softplus(energy - margin_pos)
    neg_loss = torch.nn.functional.softplus(margin_neg - energy)
    weights = torch.where(label > 0.5, label_strength.clamp_min(0.05), torch.ones_like(label))
    loss = torch.where(label > 0.5, pos_loss, neg_loss) * weights
    return loss.mean()


def logistic_energy_loss(energy, label, *, label_strength=None):
    label_strength = torch.ones_like(label) if label_strength is None else label_strength
    logits = -energy
    weights = torch.where(label > 0.5, label_strength.clamp_min(0.05), torch.ones_like(label))
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, label, weight=weights)


def gradient_penalty(energy, coordinates, *, target_norm: float = 0.0):
    grad = torch.autograd.grad(energy.sum(), coordinates, create_graph=True, retain_graph=True)[0]
    return ((grad.norm(dim=-1) - target_norm) ** 2).mean()


def binary_auc(labels, scores) -> float:
    labels_list = [int(value) for value in labels]
    scores_list = [float(value) for value in scores]
    positives = [(score, label) for score, label in zip(scores_list, labels_list) if label == 1]
    negatives = [(score, label) for score, label in zip(scores_list, labels_list) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            total += 1
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def ranking_accuracy(labels, energies) -> float:
    labels_list = [int(value) for value in labels]
    energies_list = [float(value) for value in energies]
    positive_energies = [energy for energy, label in zip(energies_list, labels_list) if label == 1]
    negative_energies = [energy for energy, label in zip(energies_list, labels_list) if label == 0]
    if not positive_energies or not negative_energies:
        return float("nan")
    correct = sum(1 for pos in positive_energies for neg in negative_energies if pos < neg)
    total = len(positive_energies) * len(negative_energies)
    return correct / total
