"""Small dependency-free geometry helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]


def as_vec3(value: Sequence[float], field_name: str = "vector") -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values")
    return (float(value[0]), float(value[1]), float(value[2]))


def add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def subtract(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def scale(a: Sequence[float], factor: float) -> Vec3:
    return (float(a[0]) * factor, float(a[1]) * factor, float(a[2]) * factor)


def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return dx * dx + dy * dy + dz * dz


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(squared_distance(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2)


def centroid(points: Iterable[Sequence[float]]) -> Vec3:
    total = [0.0, 0.0, 0.0]
    count = 0
    for point in points:
        total[0] += float(point[0])
        total[1] += float(point[1])
        total[2] += float(point[2])
        count += 1
    if count == 0:
        raise ValueError("cannot compute centroid of zero points")
    return (total[0] / count, total[1] / count, total[2] / count)


def gaussian_weight(distance_value: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-(distance_value * distance_value) / (2.0 * sigma * sigma))


def min_distance(point: Sequence[float], points: Iterable[Sequence[float]]) -> float | None:
    best: float | None = None
    for other in points:
        value = distance(point, other)
        if best is None or value < best:
            best = value
    return best

