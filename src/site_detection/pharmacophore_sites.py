"""Pharmacophore feature extraction from reference ligands.

Extracts standard pharmacophore feature points (donor, acceptor, hydrophobic,
aromatic, positive/negative ionizable) from a reference ligand SDF using RDKit's
built-in chemical feature detection. Output is a SiteMap compatible with the
existing SiteCompatibilityEnergy infrastructure.

Pharmacophore feature families → site_type mapping:
    Donor      → hbd
    Acceptor   → hba
    Hydrophobe → hydrophobic
    Aromatic   → aromatic
    PosIonizable → pos_ion
    NegIonizable → neg_ion
    ZnBinder   → (skipped — not a standard pharmacophore feature)
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

_PHARM_FAMILY_TO_TYPE = {
    "Donor": "hbd",
    "Acceptor": "hba",
    "Hydrophobe": "hydrophobic",
    "Aromatic": "aromatic",
    "PosIonizable": "pos_ion",
    "NegIonizable": "neg_ion",
}

_PHARM_TYPE_ORDER = ["hbd", "hba", "hydrophobic", "aromatic", "pos_ion", "neg_ion"]


def _build_feature_factory():
    fdef_path = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(fdef_path))


def extract_pharmacophore_sites(ref_ligand_sdf: str, protein_id: str = "",
                                  ligand_id: str = "") -> dict:
    """Extract pharmacophore feature points from a reference ligand.

    Args:
        ref_ligand_sdf: Path to reference ligand SDF file.
        protein_id: Optional protein identifier for the SiteMap.
        ligand_id: Optional ligand identifier.

    Returns:
        SiteMap dict compatible with site_schema.SiteMap, with sites keyed by
        pharmacophore feature type (hbd, hba, hydrophobic, aromatic, pos_ion,
        neg_ion). Each site has centre, radius=1.5, confidence=1.0.
    """
    factory = _build_feature_factory()
    supplier = Chem.SDMolSupplier(ref_ligand_sdf)
    mol = supplier[0]
    if mol is None:
        raise ValueError(f"Could not read ligand from {ref_ligand_sdf}")

    mol_3d = Chem.AddHs(mol)
    feats = factory.GetFeaturesForMol(mol_3d)

    sites = []
    site_id = 0
    for f in feats:
        family = f.GetFamily()
        site_type = _PHARM_FAMILY_TO_TYPE.get(family)
        if site_type is None:
            continue  # skip ZnBinder, LumpedHydrophobe etc.

        pos = f.GetPos()
        center = [float(pos.x), float(pos.y), float(pos.z)]

        sites.append({
            "site_id": site_id,
            "site_type": site_type,
            "center": center,
            "radius": 1.5,
            "score": 1.0,
            "confidence": 1.0,
            "source": "rdkit_pharmacophore",
            "features": {
                "rdkit_family": family,
            },
        })
        site_id += 1

    return {
        "schema_version": "1.0",
        "protein_id": protein_id,
        "ligand_id": ligand_id,
        "pocket_center": [0.0, 0.0, 0.0],
        "coordinate_frame": "original_pdb_coordinates",
        "sites": sites,
    }


def get_pharmacophore_type_index(site_type: str) -> int:
    """Map pharmacophore site_type string to matrix row index."""
    try:
        return _PHARM_TYPE_ORDER.index(site_type)
    except ValueError:
        return 0  # default to hbd


def get_pharmacophore_type_count() -> int:
    """Number of pharmacophore feature types."""
    return len(_PHARM_TYPE_ORDER)


def select_pharmacophore_anchors(ref_ligand_sdf: str,
                                  pharm_site_map: dict,
                                  max_anchors: int = 6) -> tuple:
    """Select anchor atoms from the reference ligand for pharmacophore guidance.

    For each pharmacophore feature site, finds the nearest heavy atom in the
    reference ligand. Returns arrays of atom indices and target positions.

    Args:
        ref_ligand_sdf: Path to reference ligand SDF.
        pharm_site_map: Pharmacophore SiteMap dict from
                        extract_pharmacophore_sites().
        max_anchors: Maximum number of anchor atoms to select.

    Returns:
        (anchor_indices: list[int], anchor_positions: np.ndarray [N,3])
    """
    supplier = Chem.SDMolSupplier(ref_ligand_sdf)
    mol = supplier[0]
    if mol is None:
        return [], np.empty((0, 3))

    mol_3d = Chem.AddHs(mol)
    conf = mol_3d.GetConformer(0)
    heavy_atom_positions = np.array([
        conf.GetAtomPosition(i) for i in range(mol_3d.GetNumAtoms())
        if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1  # non-H
    ])
    heavy_atom_indices = [
        i for i in range(mol_3d.GetNumAtoms())
        if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]

    selected_anchors = []
    used_atoms = set()

    for site in pharm_site_map.get("sites", []):
        if len(selected_anchors) >= max_anchors:
            break

        center = np.array(site["center"])
        # Find nearest unused heavy atom
        best_dist = float("inf")
        best_idx = -1
        for j, (pos, atom_idx) in enumerate(zip(heavy_atom_positions,
                                                  heavy_atom_indices)):
            if atom_idx in used_atoms:
                continue
            dist = np.linalg.norm(pos - center)
            if dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx >= 0 and best_dist < 3.0:
            selected_anchors.append(heavy_atom_indices[best_idx])
            used_atoms.add(heavy_atom_indices[best_idx])

    anchor_positions = np.array([
        conf.GetAtomPosition(i) for i in selected_anchors
    ])

    return selected_anchors, anchor_positions
