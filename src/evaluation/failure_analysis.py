"""Summarize common ESField failure patterns from metric tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def analyze_site_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n_rows": 0, "warnings": ["no metric rows provided"]}
    warnings: list[str] = []
    sor = [_float(row.get("site_occupancy_rate")) for row in rows]
    casmr = [_float(row.get("correct_atom_site_matching_rate")) for row in rows]
    swpp = [_float(row.get("stable_water_preservation_penalty")) for row in rows]
    if _mean(sor) < 0.2:
        warnings.append("SOR 均值低于 0.2，说明位点占据不足或 site map 过于严格。")
    if _mean(casmr) < 0.5:
        warnings.append("CASMR 均值低于 0.5，说明生成原子类型与位点类型匹配不足。")
    if _mean(swpp) > 0.2:
        warnings.append("SWPP 均值高于 0.2，stable water 可能被不兼容原子占据。")
    return {
        "n_rows": len(rows),
        "mean_site_occupancy_rate": _mean(sor),
        "mean_correct_atom_site_matching_rate": _mean(casmr),
        "mean_stable_water_preservation_penalty": _mean(swpp),
        "warnings": warnings,
    }


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    valid = [value for value in values if value == value]
    return sum(valid) / len(valid) if valid else float("nan")


def _read_rows(path: str | Path) -> list[dict]:
    target = Path(path)
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else [payload]
    with target.open("rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ESField site metric failure patterns.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    summary = analyze_site_metric_rows(_read_rows(args.metrics))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
