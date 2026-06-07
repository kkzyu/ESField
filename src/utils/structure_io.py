"""Minimal structure readers for PDB, SDF, and gzipped SDF files."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from .chemistry import (
    atomic_number,
    element_from_pdb_atom_name,
    infer_atom_type,
    is_hydrogen,
    normalize_element,
)


@dataclass(frozen=True)
class StructureAtom:
    index: int
    name: str
    element: str
    coord: tuple[float, float, float]
    residue_name: str = ""
    residue_id: str = ""
    chain_id: str = ""
    record_name: str = "ATOM"

    @property
    def atomic_number(self) -> int:
        return atomic_number(self.element)

    @property
    def atom_type(self) -> str:
        return infer_atom_type(self.element, self.name)


def open_text(path: str | Path) -> TextIO:
    target = Path(path)
    if target.suffix == ".gz":
        return gzip.open(target, "rt", encoding="utf-8", errors="replace")
    return target.open("rt", encoding="utf-8", errors="replace")


def read_pdb_atoms(path: str | Path, *, include_hydrogen: bool = False) -> list[StructureAtom]:
    atoms: list[StructureAtom] = []
    with open_text(path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21].strip()
                residue_id = line[22:27].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = normalize_element(line[76:78].strip() or element_from_pdb_atom_name(name))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"invalid PDB atom line in {path}: {line.rstrip()}") from exc
            if not include_hydrogen and is_hydrogen(element):
                continue
            atoms.append(
                StructureAtom(
                    index=len(atoms),
                    name=name,
                    element=element,
                    coord=(x, y, z),
                    residue_name=residue_name,
                    residue_id=residue_id,
                    chain_id=chain_id,
                    record_name=line[:6].strip(),
                )
            )
    return atoms


def read_sdf_atoms(
    path: str | Path,
    *,
    molecule_index: int = 0,
    include_hydrogen: bool = False,
) -> list[StructureAtom]:
    blocks = _iter_sdf_blocks(path)
    try:
        block = next(block for idx, block in enumerate(blocks) if idx == molecule_index)
    except StopIteration as exc:
        raise ValueError(f"molecule_index {molecule_index} not found in {path}") from exc
    return _parse_sdf_block_atoms(block, path, include_hydrogen=include_hydrogen)


def read_ligand_atoms(path: str | Path, *, include_hydrogen: bool = False) -> list[StructureAtom]:
    target = Path(path)
    lower_name = target.name.lower()
    if lower_name.endswith(".sdf") or lower_name.endswith(".sdf.gz"):
        return read_sdf_atoms(target, include_hydrogen=include_hydrogen)
    if lower_name.endswith(".pdb") or lower_name.endswith(".pdb.gz"):
        return read_pdb_atoms(target, include_hydrogen=include_hydrogen)
    raise ValueError(f"unsupported ligand file type: {target}")


def _iter_sdf_blocks(path: str | Path) -> Iterator[list[str]]:
    with open_text(path) as handle:
        block: list[str] = []
        for line in handle:
            if line.strip() == "$$$$":
                if block:
                    yield block
                    block = []
                continue
            block.append(line.rstrip("\n"))
        if block:
            yield block


def _parse_sdf_block_atoms(
    block: list[str],
    path: str | Path,
    *,
    include_hydrogen: bool,
) -> list[StructureAtom]:
    if len(block) < 4:
        raise ValueError(f"invalid SDF block in {path}: missing counts line")
    counts = block[3]
    try:
        atom_count = int(counts[0:3])
    except ValueError:
        pieces = counts.split()
        if not pieces:
            raise ValueError(f"invalid SDF counts line in {path}: {counts!r}")
        atom_count = int(pieces[0])
    atoms: list[StructureAtom] = []
    for raw_index, line in enumerate(block[4 : 4 + atom_count]):
        pieces = line.split()
        if len(pieces) < 4:
            raise ValueError(f"invalid SDF atom line in {path}: {line!r}")
        x, y, z = float(pieces[0]), float(pieces[1]), float(pieces[2])
        element = normalize_element(pieces[3])
        if not include_hydrogen and is_hydrogen(element):
            continue
        atoms.append(
            StructureAtom(
                index=len(atoms),
                name=f"{element}{raw_index + 1}",
                element=element,
                coord=(x, y, z),
                residue_name="LIG",
                residue_id="1",
                record_name="HETATM",
            )
        )
    return atoms


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    import json

    rows: list[dict] = []
    with Path(path).open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows
