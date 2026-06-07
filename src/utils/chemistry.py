"""Lightweight chemistry conventions used by ESField MVP code."""

from __future__ import annotations

from typing import Iterable

ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
    "BR": 35,
    "I": 53,
}

PDB_WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "DOD", "TIP", "SOL"})
HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
POLAR_ELEMENTS = frozenset({"N", "O", "P", "S"})
HALOGEN_ELEMENTS = frozenset({"F", "CL", "BR", "I"})
HYDROPHOBIC_ELEMENTS = frozenset({"C", "S", "F", "CL", "BR", "I"})


def normalize_element(element: str) -> str:
    token = "".join(ch for ch in element.strip().upper() if ch.isalpha())
    if not token:
        return "X"
    if len(token) >= 2 and token[:2] in ATOMIC_NUMBERS:
        return token[:2]
    return token[0]


def element_from_pdb_atom_name(atom_name: str) -> str:
    stripped = atom_name.strip()
    if not stripped:
        return "X"
    if len(stripped) >= 2 and stripped[:2].upper() in ATOMIC_NUMBERS:
        return normalize_element(stripped[:2])
    return normalize_element(stripped[0])


def atomic_number(element: str) -> int:
    return ATOMIC_NUMBERS.get(normalize_element(element), 0)


def infer_atom_type(element: str, atom_name: str = "", aromatic: bool | None = None) -> str:
    symbol = normalize_element(element)
    name = atom_name.upper()
    if symbol == "C":
        return "C_aromatic" if aromatic or "AR" in name else "C_sp3"
    if symbol == "N":
        return "N_donor" if "H" in name or name.startswith("N") else "N_acceptor"
    if symbol == "O":
        return "O_acceptor"
    if symbol == "S":
        return "S"
    if symbol in HALOGEN_ELEMENTS:
        return "halogen"
    if symbol == "P":
        return "P"
    return symbol


def is_hydrogen(element: str) -> bool:
    return normalize_element(element) in HYDROGEN_ELEMENTS


def is_water_residue(residue_name: str) -> bool:
    return residue_name.strip().upper() in PDB_WATER_RESNAMES


def is_polar_element(element: str) -> bool:
    return normalize_element(element) in POLAR_ELEMENTS


def is_hydrophobic_element(element: str) -> bool:
    return normalize_element(element) in HYDROPHOBIC_ELEMENTS


def compatible_atom_types(site_type: str) -> frozenset[str]:
    if site_type == "hydrophobic_cavity":
        return frozenset({"C_sp3", "C_aromatic", "halogen", "S"})
    if site_type == "high_energy_water":
        return frozenset({"C_sp3", "C_aromatic", "halogen", "S", "O_acceptor", "N_donor", "N_acceptor"})
    if site_type == "stable_water":
        return frozenset({"O_acceptor", "N_donor", "N_acceptor"})
    return frozenset()


def is_compatible_atom_site(atom_type: str, atomic_number_value: int, site_type: str) -> bool:
    if atom_type in compatible_atom_types(site_type):
        return True
    if site_type == "hydrophobic_cavity" and atomic_number_value in {6, 9, 16, 17, 35, 53}:
        return True
    if site_type in {"high_energy_water", "stable_water"} and atomic_number_value in {7, 8, 15, 16}:
        return True
    return False


def incompatible_atom_type_for_site(site_type: str) -> tuple[str, int]:
    if site_type == "hydrophobic_cavity":
        return ("O_acceptor", 8)
    if site_type == "stable_water":
        return ("C_sp3", 6)
    if site_type == "high_energy_water":
        return ("charged", 6)
    return ("X", 0)


def atom_type_index(atom_type: str, vocab: Iterable[str]) -> int:
    values = list(vocab)
    try:
        return values.index(atom_type)
    except ValueError:
        return values.index("unknown") if "unknown" in values else 0

