"""Atom feature vocabularies for compatibility potential training."""

ATOM_TYPE_VOCAB: tuple[str, ...] = (
    "unknown",
    "C_sp3",
    "C_aromatic",
    "N_donor",
    "N_acceptor",
    "O_acceptor",
    "S",
    "P",
    "halogen",
    "charged",
    "B",
)


def atom_type_to_index(atom_type: str) -> int:
    try:
        return ATOM_TYPE_VOCAB.index(atom_type)
    except ValueError:
        return 0

