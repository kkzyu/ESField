"""Atom type mapping between ESField and PAFlow type systems."""
from __future__ import annotations

# ESField ATOM_TYPE_VOCAB:
#   0: unknown, 1: C_sp3, 2: C_aromatic, 3: N_donor, 4: N_acceptor,
#   5: O_acceptor, 6: S, 7: P, 8: halogen, 9: charged, 10: B

# PAFlow MAP_ATOM_TYPE_AROMATIC_TO_INDEX (default mode, 13 types):
#   (1,False)=0 (1,True not in vocab), (6,False)=1, (6,True)=2, (7,False)=3, (7,True)=4,
#   (8,False)=5, (8,True)=6, (9,False)=7, (15,False)=8, (15,True)=9,
#   (16,False)=10, (16,True)=11, (17,False)=12

# PAFlow num_classes = 13 (add_aromatic mode, the default in the config)

PAFLOW_NUM_CLASSES = 13

# ESField atom_type -> PAFlow (atomic_number, is_aromatic) -> PAFlow index
# When aromatic info is unavailable, default to non-aromatic.
ESFIELD_TO_PAFLOW: dict[str, tuple[int, bool]] = {
    "unknown":      (6, False),   # fallback to carbon
    "C_sp3":        (6, False),
    "C_aromatic":   (6, True),
    "N_donor":      (7, False),
    "N_acceptor":   (7, False),
    "O_acceptor":   (8, False),
    "S":            (16, False),
    "P":            (15, False),
    "halogen":      (9, False),   # F is the most common halogen in ligands
    "charged":      (7, False),   # fallback to nitrogen
    "B":            (6, False),   # fallback to carbon
}

PAFLOW_AROMATIC_INDEX = {
    (1, False): 0,
    (6, False): 1,  (6, True): 2,
    (7, False): 3,  (7, True): 4,
    (8, False): 5,  (8, True): 6,
    (9, False): 7,
    (15, False): 8, (15, True): 9,
    (16, False): 10, (16, True): 11,
    (17, False): 12,
}

PAFLOW_INDEX_TO_ATOMIC = {
    0: 1, 1: 6, 2: 6, 3: 7, 4: 7,
    5: 8, 6: 8, 7: 9, 8: 15, 9: 15,
    10: 16, 11: 16, 12: 17,
}


def esfield_atom_type_to_paflow_index(atom_type: str) -> int:
    atomic_num, is_aromatic = ESFIELD_TO_PAFLOW.get(atom_type, (6, False))
    return PAFLOW_AROMATIC_INDEX.get((atomic_num, is_aromatic), 1)


def esfield_atom_types_to_paflow_indices(atom_types: list[str]) -> list[int]:
    return [esfield_atom_type_to_paflow_index(t) for t in atom_types]


def paflow_index_to_atomic_number(paflow_index: int) -> int:
    return PAFLOW_INDEX_TO_ATOMIC.get(paflow_index, 6)


def expect_paflow_model() -> str:
    return (
        "PAFlow 是 pocket-conditioned 3D flow matching 生成器 (NeurIPS 2025)。"
        "ESField 采样期 site-aware energy guidance 通过修改 ODE velocity field 注入。"
        "此 adapter 提供原子类型映射和 guidance 注入点信息。"
    )
