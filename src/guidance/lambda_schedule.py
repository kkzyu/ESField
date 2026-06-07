"""Guidance strength schedules."""

from __future__ import annotations

import math


def guidance_lambda(
    t: float,
    *,
    lambda_max: float = 0.1,
    guidance_start: float = 0.35,
    guidance_end: float = 0.90,
    schedule: str = "sigmoid",
) -> float:
    if t < guidance_start or t > guidance_end or lambda_max <= 0:
        return 0.0
    span = max(guidance_end - guidance_start, 1.0e-8)
    x = (t - guidance_start) / span
    if schedule == "linear":
        return lambda_max * x
    if schedule == "cosine":
        return lambda_max * 0.5 * (1.0 - math.cos(math.pi * x))
    if schedule == "constant":
        return lambda_max
    if schedule == "sigmoid":
        return lambda_max / (1.0 + math.exp(-12.0 * (x - 0.5)))
    raise ValueError(f"unknown guidance schedule: {schedule}")

