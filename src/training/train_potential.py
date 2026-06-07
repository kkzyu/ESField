"""Train the ESField atom-site compatibility potential."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from models.losses import binary_auc, logistic_energy_loss, margin_energy_loss, ranking_accuracy
from models.potential_network import CompatibilityPotential, PotentialConfig, tensor_batch_from_pairs
from training.dataset import AtomSitePairDataset
from models.distance_encoding import require_torch

torch = require_torch()


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = AtomSitePairDataset(args.train_pairs)
    valid_dataset = AtomSitePairDataset(args.valid_pairs) if args.valid_pairs else None
    if len(train_dataset) == 0:
        raise ValueError("train pair file is empty")

    config = PotentialConfig(
        atom_embed_dim=args.atom_embed_dim,
        site_embed_dim=args.site_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cutoff=args.cutoff,
        energy_clip=args.energy_clip,
    )
    model = CompatibilityPotential(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(output_dir, args, config, device)

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(len(train_dataset), generator=rng).tolist()
        losses: list[float] = []
        for start in range(0, len(indices), args.batch_size):
            batch_pairs = [train_dataset.pairs[idx] for idx in indices[start : start + args.batch_size]]
            batch = _move_batch(tensor_batch_from_pairs(batch_pairs), device)
            optimizer.zero_grad(set_to_none=True)
            energy = model(
                batch["atom_type_idx"],
                batch["site_type_idx"],
                batch["relative_position"],
                batch["distance"],
                batch["site_radius"],
                batch["site_confidence"],
            )
            if args.loss == "logistic":
                loss = logistic_energy_loss(energy, batch["label"], label_strength=batch["label_strength"])
            else:
                loss = margin_energy_loss(
                    energy,
                    batch["label"],
                    label_strength=batch["label_strength"],
                    margin_pos=args.margin_pos,
                    margin_neg=args.margin_neg,
                )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_metrics = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1)}
        if valid_dataset is not None and len(valid_dataset) > 0:
            train_metrics.update(_evaluate(model, valid_dataset.pairs, device, args.batch_size))
        history.append(train_metrics)
        _write_history(output_dir / "metrics.csv", history)
        print(json.dumps(train_metrics, ensure_ascii=False))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(config), "epoch": epoch, "metrics": train_metrics},
                output_dir / f"compatibility_potential_epoch_{epoch:04d}.pt",
            )

    return history[-1]


def _evaluate(model, pairs, device, batch_size: int) -> dict:
    model.eval()
    labels: list[int] = []
    energies: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            batch = _move_batch(tensor_batch_from_pairs(batch_pairs), device)
            energy = model(
                batch["atom_type_idx"],
                batch["site_type_idx"],
                batch["relative_position"],
                batch["distance"],
                batch["site_radius"],
                batch["site_confidence"],
            )
            labels.extend(int(value) for value in batch["label"].detach().cpu().tolist())
            energies.extend(float(value) for value in energy.detach().cpu().tolist())
    compatibility_scores = [-value for value in energies]
    return {
        "valid_auc": binary_auc(labels, compatibility_scores),
        "valid_ranking_accuracy": ranking_accuracy(labels, energies),
        "valid_mean_pos_energy": _mean([energy for energy, label in zip(energies, labels) if label == 1]),
        "valid_mean_neg_energy": _mean([energy for energy, label in zip(energies, labels) if label == 0]),
    }


def _move_batch(batch: dict, device):
    return {key: value.to(device) for key, value in batch.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _write_history(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(output_dir: Path, args: argparse.Namespace, config: PotentialConfig, device) -> None:
    metadata = {
        "args": vars(args),
        "model_config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESField compatibility potential.")
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--valid-pairs", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--loss", choices=["margin", "logistic"], default="margin")
    parser.add_argument("--margin-pos", type=float, default=-1.0)
    parser.add_argument("--margin-neg", type=float, default=1.0)
    parser.add_argument("--atom-embed-dim", type=int, default=32)
    parser.add_argument("--site-embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--energy-clip", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train(parse_args())


if __name__ == "__main__":
    main()
