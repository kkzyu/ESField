"""Generate a static HTML report from ESField result tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def generate_html_report(
    *,
    output: str | Path,
    title: str,
    metric_files: list[str | Path],
    notes: str = "",
) -> None:
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;margin:16px 0;width:100%}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}th{background:#f3f5f7}.warn{color:#9a3412}</style>",
        f"<title>{html.escape(title)}</title></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    if notes:
        sections.append(f"<p>{html.escape(notes)}</p>")
    for metric_file in metric_files:
        rows = _read_rows(metric_file)
        sections.append(f"<h2>{html.escape(str(metric_file))}</h2>")
        sections.append(_rows_to_html(rows))
    sections.append("</body></html>")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(sections), encoding="utf-8")


def _read_rows(path: str | Path) -> list[dict]:
    target = Path(path)
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else [payload]
    with target.open("rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_to_html(rows: list[dict]) -> str:
    if not rows:
        return "<p class='warn'>No rows.</p>"
    keys = sorted({key for row in rows for key in row.keys()})
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(key)}</th>" for key in keys)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        out.extend(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ESField HTML result report.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="ESField MVP Report")
    parser.add_argument("--metric-files", nargs="+", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    generate_html_report(output=args.output, title=args.title, metric_files=args.metric_files, notes=args.notes)
    print(f"wrote report: {args.output}")


if __name__ == "__main__":
    main()
