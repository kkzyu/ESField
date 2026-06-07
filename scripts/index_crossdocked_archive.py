"""Safely list or sample-extract CrossDocked archive members."""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a CrossDocked tar/tgz without full extraction.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--extract-dir", default=None)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    archive = Path(args.archive)
    selected_members = []
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            selected_members.append(member)
            rows.append(
                {
                    "name": member.name,
                    "size": member.size,
                    "kind": _classify_member(member.name),
                }
            )
            if len(rows) >= args.limit:
                break
        if args.extract:
            if not args.extract_dir:
                raise ValueError("--extract requires --extract-dir")
            target = Path(args.extract_dir).resolve()
            if args.dry_run:
                print(f"dry-run: would extract {len(rows)} members to {target}")
            else:
                target.mkdir(parents=True, exist_ok=True)
                safe_members = [member for member in selected_members if _is_safe_member(target, member.name)]
                tar.extractall(target, members=safe_members)
                print(f"extracted {len(safe_members)} members to {target}")

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "size", "kind"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"indexed {len(rows)} archive members: {output}")


def _classify_member(name: str) -> str:
    lower = name.lower()
    if lower.endswith("_rec.pdb"):
        return "receptor_pdb"
    if lower.endswith("_lig.pdb"):
        return "ligand_pdb"
    if lower.endswith(".sdf") or lower.endswith(".sdf.gz"):
        return "ligand_or_docked_sdf"
    if lower.endswith(".gninatypes"):
        return "gninatypes"
    if lower.endswith(".types"):
        return "types"
    return "other"


def _is_safe_member(root: Path, name: str) -> bool:
    resolved = (root / name).resolve()
    return root == resolved or root in resolved.parents


if __name__ == "__main__":
    main()
