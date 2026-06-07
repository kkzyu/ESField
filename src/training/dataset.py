"""PyTorch dataset for atom-site pair JSONL files."""

from __future__ import annotations

from data.atom_site_schema import read_pairs_jsonl
from models.potential_network import tensor_batch_from_pairs
from models.distance_encoding import require_torch

torch = require_torch()


class AtomSitePairDataset(torch.utils.data.Dataset):
    def __init__(self, path: str) -> None:
        self.pairs = read_pairs_jsonl(path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        return self.pairs[index]


def collate_atom_site_pairs(pairs):
    return tensor_batch_from_pairs(pairs)

