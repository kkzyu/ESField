#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text) or {}
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML config files.")
        return yaml.safe_load(text) or {}
    raise ValueError("Config file must be .json, .yaml, or .yml.")


def _coalesce(*values: Optional[Any]) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def _read_prefixes_file(path: Path) -> List[str]:
    prefixes: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        prefixes.append(cleaned)
    return prefixes


def _normalize_name(name: str) -> str:
    return name.lstrip("./")


def _member_prefix(name: str) -> str:
    normalized = _normalize_name(name)
    if not normalized:
        return ""
    return normalized.split("/", 1)[0]


def _collect_prefixes(archive_path: Path, limit: int) -> List[str]:
    prefixes: List[str] = []
    seen: Set[str] = set()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            prefix = _member_prefix(member.name)
            if not prefix or prefix in seen:
                continue
            prefixes.append(prefix)
            seen.add(prefix)
            if limit > 0 and len(prefixes) >= limit:
                break
    return prefixes


def _is_within_directory(base_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def _is_selected(member: tarfile.TarInfo, selected_prefixes: Optional[Set[str]]) -> bool:
    if selected_prefixes is None:
        return True
    prefix = _member_prefix(member.name)
    return bool(prefix) and prefix in selected_prefixes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partially extract CrossDocked archives by top-level prefixes."
    )
    parser.add_argument("--config", type=Path, help="JSON/YAML config file.")
    parser.add_argument("--archive", type=Path, help="Path to .tgz archive.")
    parser.add_argument("--output", type=Path, help="Output directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of top-level prefixes to extract.",
    )
    parser.add_argument(
        "--prefixes-file",
        type=Path,
        help="Text file containing top-level prefixes to extract.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List only, no extraction.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Log progress every N extracted members.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config: Dict[str, Any] = {}
    if args.config:
        config = _load_config(args.config)
        if isinstance(config.get("extract"), dict):
            config = config["extract"]

    archive_value = _coalesce(args.archive, config.get("archive"), config.get("archive_path"))
    output_value = _coalesce(args.output, config.get("output"), config.get("output_dir"))

    if archive_value is None:
        raise RuntimeError("archive path is required (via --archive or config).")
    if output_value is None:
        raise RuntimeError("output directory is required (via --output or config).")

    archive_path = Path(archive_value).expanduser()
    output_dir = Path(output_value).expanduser()

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    limit = args.limit if args.limit is not None else int(config.get("limit", 0))
    dry_run = bool(config.get("dry_run", False)) or args.dry_run
    overwrite = bool(config.get("overwrite", False)) or args.overwrite
    log_every = args.log_every if args.log_every is not None else int(config.get("log_every", 200))

    prefixes_file_value = _coalesce(args.prefixes_file, config.get("prefixes_file"))
    prefixes: Optional[List[str]] = None
    if prefixes_file_value:
        prefixes = _read_prefixes_file(Path(prefixes_file_value).expanduser())
    elif limit > 0:
        prefixes = _collect_prefixes(archive_path, limit)

    selected_prefixes: Optional[Set[str]] = None
    if prefixes:
        selected_prefixes = set(prefixes)
        preview = ", ".join(prefixes[:10])
        print(f"Selected {len(prefixes)} prefixes: {preview}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    selected_count = 0
    skipped_existing = 0
    skipped_links = 0

    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if not _is_selected(member, selected_prefixes):
                continue

            if member.issym() or member.islnk():
                skipped_links += 1
                continue

            target_path = output_dir / member.name
            if not _is_within_directory(output_dir, target_path):
                raise RuntimeError(f"Unsafe member path: {member.name}")

            if not overwrite and member.isfile() and target_path.exists():
                skipped_existing += 1
                continue

            selected_count += 1

            if dry_run:
                continue

            archive.extract(member, path=output_dir)
            extracted_count += 1

            if log_every > 0 and extracted_count % log_every == 0:
                print(f"Extracted {extracted_count} members...")

    print(f"Members selected: {selected_count}")
    if dry_run:
        print("Dry-run only. No files were extracted.")
    else:
        print(
            "Extraction done. "
            f"Extracted={extracted_count}, skipped_existing={skipped_existing}, "
            f"skipped_links={skipped_links}."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
