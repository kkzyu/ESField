#!/usr/bin/env python3
"""Extract pharmacophore feature points from a reference ligand.

Identifies up to ~11 pharmacophore features:
  - HBD (H-bond donor): N-H, O-H
  - HBA (H-bond acceptor): O, N with lone pairs
  - Hydrophobic: contiguous apolar atom clusters
  - Aromatic: aromatic ring centroids
  - PosIonizable: protonatable amines
  - NegIonizable: carboxylate groups

Stores features as a pharmacophore site map in the same JSON format as HEW sites,
compatible with SiteCompatibilityEnergy.

Usage:
    python scripts/extract_pharmacophore.py \
        --ligand /path/to/ligand.sdf \
        --output /path/to/pharm_sites.json
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D


def extract_pharmacophore_features(mol, conf_id=0):
    """Extract pharmacophore feature points from a molecule.

    Returns list of dicts with keys: center (x,y,z), pharm_type, radius, confidence.
    """
    if mol is None:
        return []

    conf = mol.GetConformer(conf_id)
    if conf is None:
        return []

    features = []

    # ── 1. H-bond Donors: N-H, O-H groups ──
    hbd_smarts = [
        "[$([N;!H0;v3,v4&+1]),$([O,S;H1;v2]);!$(*=[N,O,P,S])]",  # standard
        "[N&!H0&v3]",  # amine N-H
        "[O&H1&v2]",   # hydroxyl
    ]
    for smarts in hbd_smarts:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                pos = np.array(conf.GetAtomPosition(idx))
                # Use H atom position (extended along bond)
                features.append({
                    "center": pos.tolist(),
                    "pharm_type": "hbd",
                    "radius": 1.2,
                    "confidence": 1.0,
                    "source_atom_idx": idx,
                })
                break  # one per match to avoid duplicates

    # ── 2. H-bond Acceptors: O, N with lone pairs ──
    hba_smarts = [
        "[$([O;H0;v2]),$([O;H1;v2;!$(O-N)]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([n;+0])]",
        "[O&H0&v2]",  # carbonyl
        "[n;+0]",     # aromatic N
    ]
    for smarts in hba_smarts:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() in (7, 8):
                    pos = np.array(conf.GetAtomPosition(idx))
                    features.append({
                        "center": pos.tolist(),
                        "pharm_type": "hba",
                        "radius": 1.2,
                        "confidence": 1.0,
                        "source_atom_idx": idx,
                    })
                    break

    # ── 3. Hydrophobic: non-polar atoms ──
    hydrophobic_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
            # aliphatic carbon = hydrophobic
            hydrophobic_atoms.append(atom.GetIdx())
        elif atom.GetAtomicNum() in (17, 35, 53, 9):  # halogens
            hydrophobic_atoms.append(atom.GetIdx())

    # Cluster contiguous hydrophobic atoms
    if hydrophobic_atoms:
        # Simple clustering: find connected components
        visited = set()
        clusters = []
        for idx in hydrophobic_atoms:
            if idx in visited:
                continue
            cluster = []
            stack = [idx]
            while stack:
                i = stack.pop()
                if i in visited:
                    continue
                visited.add(i)
                cluster.append(i)
                atom = mol.GetAtomWithIdx(i)
                for nb in atom.GetNeighbors():
                    if nb.GetIdx() in hydrophobic_atoms and nb.GetIdx() not in visited:
                        stack.append(nb.GetIdx())
            if len(cluster) >= 2:  # at least 2 contiguous
                clusters.append(cluster)

        for cluster in clusters[:3]:  # top 3 clusters
            coords = np.array([conf.GetAtomPosition(i) for i in cluster])
            centroid = coords.mean(axis=0)
            features.append({
                "center": centroid.tolist(),
                "pharm_type": "hydrophobic",
                "radius": 2.0,
                "confidence": min(1.0, len(cluster) / 5.0),
                "source_atom_indices": cluster,
            })

    # ── 4. Aromatic rings ──
    rings = mol.GetRingInfo().AtomRings()
    aromatic_rings = []
    for ring in rings:
        if len(ring) < 5:
            continue
        all_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        if all_aromatic:
            coords = np.array([conf.GetAtomPosition(i) for i in ring])
            centroid = coords.mean(axis=0)
            aromatic_rings.append(centroid)

    for i, centroid in enumerate(aromatic_rings[:2]):  # top 2
        features.append({
            "center": centroid.tolist(),
            "pharm_type": "aromatic",
            "radius": 1.5,
            "confidence": 1.0,
            "ring_index": i,
        })

    # ── 5. PosIonizable: protonatable amines ──
    pos_smarts = ["[N;+0;!$(N-C=O)]", "[N;H0;+0]", "[NH2;+0]"]
    for smarts in pos_smarts:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches[:2]:  # top 2
            idx = match[0]
            pos = np.array(conf.GetAtomPosition(idx))
            features.append({
                "center": pos.tolist(),
                "pharm_type": "pos_ion",
                "radius": 1.5,
                "confidence": 0.8,
                "source_atom_idx": idx,
            })

    # ── 6. NegIonizable: carboxylate groups ──
    neg_smarts = ["C(=O)[O-]", "C(=O)O", "[O-]P(=O)"]
    for smarts in neg_smarts:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches[:2]:
            idx = match[0]  # central C/P
            pos = np.array(conf.GetAtomPosition(idx))
            features.append({
                "center": pos.tolist(),
                "pharm_type": "neg_ion",
                "radius": 1.5,
                "confidence": 0.8,
                "source_atom_idx": idx,
            })

    # ── Deduplicate by distance (merge features within 1.0 Å) ──
    deduped = []
    used = set()
    for i, f1 in enumerate(features):
        if i in used:
            continue
        c1 = np.array(f1["center"])
        for j, f2 in enumerate(features):
            if j <= i or j in used:
                continue
            c2 = np.array(f2["center"])
            if np.linalg.norm(c1 - c2) < 1.0 and f1["pharm_type"] == f2["pharm_type"]:
                used.add(j)
        deduped.append(f1)

    # ── Cap at ~11 features ──
    # Prioritize: hbd, hba, aromatic, hydrophobic, pos_ion, neg_ion
    priority = {"hbd": 1, "hba": 2, "aromatic": 3, "hydrophobic": 4, "neg_ion": 5, "pos_ion": 6}
    deduped.sort(key=lambda f: priority.get(f["pharm_type"], 99))
    deduped = deduped[:12]

    return deduped


def features_to_site_map(features, protein_id="unknown"):
    """Convert pharmacophore features to HEW-compatible site map JSON."""
    PHARM_TYPE_TO_ENV = {
        "hbd": "polar_unsatisfied",
        "hba": "polar_unsatisfied",
        "hydrophobic": "hydrophobic",
        "aromatic": "hydrophobic",
        "pos_ion": "mixed",
        "neg_ion": "mixed",
    }

    sites = []
    for i, feat in enumerate(features):
        sites.append({
            "site_id": i,
            "site_type": "high_energy_water",  # reuse HEW type for compatibility
            "center": feat["center"],
            "radius": feat.get("radius", 1.4),
            "score": 1.0,
            "confidence": feat.get("confidence", 1.0),
            "source": "pharmacophore",
            "pharm_type": feat["pharm_type"],
            "features": {
                "hbond_count": 2 if feat["pharm_type"] in ("hbd", "hba") else 0,
                "hydrophobic_contact_count": 6 if feat["pharm_type"] == "hydrophobic" else 3,
                "nearest_protein_distance": 3.0,
            },
        })

    # Compute pocket center
    centers = np.array([f["center"] for f in features])
    pocket_center = centers.mean(axis=0).tolist()

    return {
        "schema_version": "1.0",
        "protein_id": protein_id,
        "ligand_id": f"{protein_id}_ligand",
        "pocket_center": pocket_center,
        "coordinate_frame": "original_pdb_coordinates",
        "sites": sites,
        "pharmacophore_types": [f["pharm_type"] for f in features],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligand", required=True, help="Reference ligand SDF")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--protein-id", default="unknown")
    args = parser.parse_args()

    mol = Chem.SDMolSupplier(args.ligand)[0]
    if mol is None:
        print("ERROR: Could not read ligand")
        sys.exit(1)

    # Add hydrogens for better HBD detection
    mol = Chem.AddHs(mol)

    features = extract_pharmacophore_features(mol)
    site_map = features_to_site_map(features, args.protein_id)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(site_map, f, indent=2)

    print(f"Extracted {len(features)} pharmacophore features:")
    for i, feat in enumerate(features):
        c = feat["center"]
        print(f"  {i}: {feat['pharm_type']:<15} at ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
