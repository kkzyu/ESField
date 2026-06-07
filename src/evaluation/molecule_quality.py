"""Basic molecule quality summaries with optional RDKit support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utils.structure_io import read_ligand_atoms


def summarize_molecule(path: str | Path) -> dict:
    atoms = read_ligand_atoms(path)
    element_counts: dict[str, int] = {}
    for atom in atoms:
        element_counts[atom.element] = element_counts.get(atom.element, 0) + 1
    row = {
        "file": str(path),
        "heavy_atom_count": len(atoms),
        "element_counts": element_counts,
        "qed": None,
        "sa_score": None,
        "rdkit_status": "not_used",
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
    except ImportError:
        row["rdkit_status"] = "rdkit_not_installed"
        return row
    mol = _load_rdkit_mol(path, Chem)
    if mol is None:
        row["rdkit_status"] = "rdkit_load_failed"
        return row
    row["rdkit_status"] = "ok"
    row["qed"] = float(QED.qed(mol))
    row["sa_score"] = _try_sa_score(mol)
    return row


def _load_rdkit_mol(path: str | Path, Chem):
    lower = str(path).lower()
    if lower.endswith(".sdf") or lower.endswith(".sdf.gz"):
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        return supplier[0] if supplier and len(supplier) else None
    if lower.endswith(".pdb"):
        return Chem.MolFromPDBFile(str(path), removeHs=False)
    return None


def _try_sa_score(mol):
    try:
        import sascorer  # type: ignore
    except ImportError:
        return None
    return float(sascorer.calculateScore(mol))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize molecule quality proxies.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    rows = [summarize_molecule(path) for path in args.inputs]
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_csv:
        target = Path(args.output_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        keys = ["file", "heavy_atom_count", "element_counts", "qed", "sa_score", "rdkit_status"]
        with target.open("wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

