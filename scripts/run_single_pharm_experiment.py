#!/usr/bin/env python3
"""Single pharmacophore point constraint experiment — KAG mechanism validation.

Tests whether KAG (CoM projection) outperforms baselines when targeting a
single, isolated pharmacophore point — analogous to a single HEW site.

4 conditions × 50 mols on 3mfw:
  - unguided:  no guidance
  - full_grad: per-atom gradient toward single HBD point
  - hard_fix:  one anchor atom fixed to nearest ref-ligand atom
  - kag:       CoM projection toward single HBD point
"""

from __future__ import annotations

import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"
DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
sys.path.insert(0, os.path.join(DRUGFLOW_DIR, "src"))
sys.path.insert(0, DRUGFLOW_DIR)

from src.model import lightning as lmod
from src.data.data_utils import process_raw_pair, TensorDict
from src.data.dataset import ProcessedLigandPocketDataset
from torch.utils.data import DataLoader
from functools import partial
from Bio.PDB import PDBParser

from guidance.latent_guidance import SiteCompatibilityEnergy, ATOM_TYPE_VOCAB, N_ATOM_TYPES
from guidance.pharmacophore_guidance import PHARMACOPHORE_COMPAT_MATRIX, PHARM_TYPE_TO_IDX
from guidance.hard_fix import HardFixCallback, patch_drugflow_hardfix, patch_drugflow_sample_post_step
from guidance.kinematic_anchor import KinematicAnchorGuidance

OUTPUT_BASE = ROOT / "results/exp_single_pharmacophore"
DATA_BASE = Path("/root/autodl-tmp/data/PDB/P-L/2001-2010/3mfw")
N_MOLS = 50


# ═══════════════════════════════════════════════════════════════════════════
# Single-Point Guidance
# ═══════════════════════════════════════════════════════════════════════════

def build_single_point_energy(point_info, sigma_distance=3.0):
    """Build SiteCompatibilityEnergy for a single pharmacophore point."""
    energy = SiteCompatibilityEnergy(
        sigma_distance=sigma_distance,
        compatibility_matrix=PHARMACOPHORE_COMPAT_MATRIX.clone(),
    )
    center = torch.tensor([point_info["center"]], dtype=torch.float32)
    type_idx = torch.tensor([PHARM_TYPE_TO_IDX[point_info["selected_pharm_type"]]], dtype=torch.long)
    conf = torch.tensor([1.0], dtype=torch.float32)
    energy.register_sites(center, type_idx, conf)
    return energy


class SinglePointGuideFn:
    """guide_log_prob for single pharmacophore point."""
    def __init__(self, energy, lambda_guide=1.0, gs=0.1, ge=0.90):
        self.energy = energy; self.lambda_guide = lambda_guide
        self.esfield_lambda = lambda_guide; self.grad_clip = 0.5
        self.guidance_start = gs; self.guidance_end = ge

    def to(self, d): self.energy.to(d); return self

    def __call__(self, t_array, *, x, h, batch_mask, bonds=None, bond_types=None):
        t = float(t_array[0] if hasattr(t_array, "__len__") else t_array)
        if t < self.guidance_start or t > self.guidance_end: return torch.tensor(0.0, device=x.device)
        hs = h.sum(dim=-1)
        ap = h if (h>=0).all() and (h<=1).all() and torch.allclose(hs, torch.ones_like(hs), atol=0.01) else F.softmax(h, dim=-1)
        return -self.lambda_guide * self.energy(x, atom_type_probs=ap)


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(device="cuda:0"):
    warnings.filterwarnings("ignore")
    _o = torch.load
    torch.load = lambda *a, **kw: _o(*a, **{**kw, "weights_only": False})
    try: m = lmod.DrugFlow.load_from_checkpoint(DRUGFLOW_CKPT, map_location=device)
    finally: torch.load = _o
    return m.to(device).eval()


def process_protein(model):
    pdb_model = PDBParser(QUIET=True).get_structure("", str(DATA_BASE / "3mfw_protein.pdb"))[0]
    rdmol = Chem.SDMolSupplier(str(DATA_BASE / "3mfw_ligand.sdf"))[0]
    ref_size = rdmol.GetNumAtoms()
    ligand_raw, pocket_raw = process_raw_pair(pdb_model, rdmol, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation, compute_nerf_params=True)
    ligand_raw["name"] = "ligand"
    collate = partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None)
    data = next(iter(DataLoader([{"ligand": ligand_raw, "pocket": pocket_raw}], batch_size=1, collate_fn=collate)))
    return data, ref_size


def gen_mols(model, protein_data, full_size, device, guide_fn=None, post_cb=None, pre_cb=None):
    patch_drugflow_hardfix()
    patch_drugflow_sample_post_step(model)
    mols = []
    t0 = time.time()
    with torch.no_grad():
        kw = dict(data=protein_data, n_samples=N_MOLS, timesteps=100, num_nodes=full_size)
        if guide_fn: kw["guide_log_prob"] = guide_fn
        if post_cb: kw["post_step_callback"] = post_cb
        if pre_cb: kw["pre_step_callback"] = pre_cb
        rdmols, _, _ = model.sample(**kw)
    for m in rdmols:
        if m is not None:
            try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except: pass
            mols.append(m)
    return mols, time.time() - t0


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(mols, point_center, pharm_type):
    """Compute all metrics for molecules against single point."""
    from metrics_new import compute_mol_centroid
    pc = np.array(point_center)
    results = []
    point_energy = build_single_point_energy(
        {"center": point_center, "selected_pharm_type": pharm_type})
    point_energy.to("cpu")

    for mol in mols:
        if mol is None or mol.GetNumAtoms() == 0: continue
        conf = mol.GetConformer()
        if conf is None: continue

        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(mol.GetNumAtoms())])
        centroid = coords.mean(axis=0)

        # min_dist to point
        dists = np.linalg.norm(coords - pc[None, :], axis=1)
        min_dist = float(dists.min())

        # Single-point COS
        sigma2 = 2.0 * 1.5**2
        gauss = np.exp(-dists**2 / sigma2)
        type_bonus = np.zeros(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            anum = atom.GetAtomicNum()
            if pharm_type == "hbd" and anum == 7: type_bonus[i] = 1.0
            elif pharm_type == "hbd" and anum == 8: type_bonus[i] = 0.5
            elif pharm_type == "hba" and anum == 8: type_bonus[i] = 1.0
            elif pharm_type == "hba" and anum == 7: type_bonus[i] = 0.5
            elif pharm_type == "hydrophobic" and anum == 6 and not atom.GetIsAromatic(): type_bonus[i] = 1.0
            elif pharm_type == "aromatic" and atom.GetIsAromatic(): type_bonus[i] = 1.0
        cos_val = float((gauss * type_bonus).max())

        # E_pharm for single point
        x_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        h_onehot = torch.zeros(1, mol.GetNumAtoms(), N_ATOM_TYPES)
        for i, atom in enumerate(mol.GetAtoms()):
            anum = atom.GetAtomicNum()
            iso = atom.GetIsAromatic()
            if anum == 6: idx = 2 if iso else 1
            elif anum == 7: idx = 3
            elif anum == 8: idx = 5
            elif anum == 16: idx = 6
            elif anum == 15: idx = 7
            elif anum in (9,17,35,53): idx = 8
            else: idx = 0
            h_onehot[0, i, idx] = 1.0
        e_val = float(point_energy(x_t[0], atom_type_probs=h_onehot[0]).cpu())

        try: qed_val = QED.qed(mol)
        except: qed_val = float("nan")
        try:
            from rdkit.Contrib.SA_Score import sascorer
            sa_val = sascorer.calculateScore(mol)
        except: sa_val = float("nan")

        results.append({
            "min_dist": min_dist, "COS": cos_val, "E_pharm": e_val,
            "QED": qed_val, "SA": sa_val,
            "centroid_dist_to_point": float(np.linalg.norm(centroid - pc)),
            "n_atoms": mol.GetNumAtoms(),
        })
    return results


def compile_stats(results):
    keys = ["min_dist", "COS", "E_pharm", "QED", "SA", "centroid_dist_to_point"]
    stats = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r.get(k, float("nan")))]
        if vals:
            stats[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    point_info = json.loads(open(OUTPUT_BASE / "3mfw_single_point.json").read())
    pc = point_info["center"]
    ptype = point_info["selected_pharm_type"]
    print(f"Single point: {ptype} at ({pc[0]:.1f}, {pc[1]:.1f}, {pc[2]:.1f})")

    print("Loading model...")
    model = load_model()
    data, ref_size = process_protein(model)
    protein_data = {"ligand": TensorDict(**data["ligand"]).to("cuda:0"),
                    "pocket": TensorDict(**data["pocket"]).to("cuda:0")}
    full_size = ref_size
    print(f"Reference size: {ref_size} atoms")

    # Get anchor atom from reference ligand (nearest to point)
    ref_mol = Chem.SDMolSupplier(str(DATA_BASE / "3mfw_ligand.sdf"))[0]
    ref_conf = ref_mol.GetConformer()
    best_d, best_pos = float("inf"), None
    for i in range(ref_mol.GetNumAtoms()):
        pos = np.array(ref_conf.GetAtomPosition(i))
        d = np.linalg.norm(pos - np.array(pc))
        if d < best_d: best_d = d; best_pos = pos
    anchor_coords = best_pos.tolist()
    print(f"Anchor atom from ref ligand at distance {best_d:.1f}Å from pharm point")

    energy = build_single_point_energy(point_info).to("cuda:0")

    for cond, guide_fn, post_cb in [
        ("unguided", None, None),
        ("full_gradient", SinglePointGuideFn(energy, lambda_guide=3.0).to("cuda:0"), None),
        ("hard_fix", SinglePointGuideFn(energy, lambda_guide=0.1).to("cuda:0"),
         HardFixCallback(anchor_indices=[0], anchor_coords=torch.tensor([anchor_coords]), fix_coords=True, fix_types=False)),
        ("kag", None,
         KinematicAnchorGuidance(anchor_indices=[0, 1], site_energy=energy,
                                 total_steps=100, lambda_max=3.0, profile="quadratic", track_kpe=True)),
    ]:
        out_dir = OUTPUT_BASE / cond
        out_dir.mkdir(parents=True, exist_ok=True)
        sdf_path = out_dir / "molecules.sdf"
        if sdf_path.exists() and sdf_path.stat().st_size > 1000:
            print(f"  {cond}: exists, skip")
            continue

        print(f"  {cond} ({N_MOLS} mols)...", end=" ", flush=True)
        mols, t = gen_mols(model, protein_data, full_size, "cuda:0", guide_fn, post_cb)

        w = Chem.SDWriter(str(sdf_path)); w.SetKekulize(False)
        for m in mols:
            try: w.write(m)
            except: pass
        w.close()

        metrics = compute_metrics(mols, pc, ptype)
        stats = compile_stats(metrics)
        json.dump({"condition": cond, "n": len(mols), "time_s": t, "statistics": stats, "per_mol": metrics},
                  open(out_dir / "metadata.json", "w"), indent=2, default=str)

        print(f"{len(mols)} mols, {t:.0f}s")
        for k in ["min_dist","COS","E_pharm","QED","SA"]:
            s = stats.get(k, {})
            if s: print(f"    {k:<12} {s['mean']:.4f}±{s['std']:.4f}")

    print("\n✓ Single-point pharmacophore experiment complete")


if __name__ == "__main__":
    main()
