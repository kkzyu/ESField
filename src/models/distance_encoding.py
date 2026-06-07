"""Distance encodings for the compatibility potential."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised on machines without torch.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for ESField model code. Install torch before training or guidance.") from _TORCH_IMPORT_ERROR
    return torch


def rbf_encode(distances, *, num_bins: int = 16, cutoff: float = 6.0):
    torch_module = require_torch()
    centers = torch_module.linspace(0.0, cutoff, num_bins, device=distances.device, dtype=distances.dtype)
    widths = cutoff / max(num_bins - 1, 1)
    return torch_module.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2.0 * widths * widths))

