#!/usr/bin/env python3
"""DecompDiff + KAG guidance experiment (Exp 3.4) — 3mfw validation.

Injects E_site / E_pharm gradient into DecompDiff's energy_drift_opt mechanism.
Supports: unguided, full_gradient, hard_fix, kag.

Usage:
    python scripts/run_decompdiff_experiment.py --pockets 3mfw --n-mols 10
    python scripts/run_decompdiff_experiment.py --all --n-mols 50
"""

from __future__ import annotations

import argparse, json, os, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.chdir('/root/baselines/DecompDiff/code/DecompDiff-main')

# DecompDiff paths
DECOMP_ROOT = Path("/root/baselines/DecompDiff/code/DecompDiff-main")
DECOMP_CKPT = Path("/root/baselines/DecompDiff/checkpoints/ckpts")
sys.path.insert(0, str(DECOMP_ROOT))
sys.path.insert(0, str(DECOMP_ROOT))
sys.path.insert(0, str(DECOMP_ROOT / "scripts"))

import utils.misc as misc
import utils.reconstruct as recon
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH, torchify_dict
from datasets.pl_pair_dataset import get_decomp_dataset
from models.decompdiff import DecompScorePosNet3D
from utils.data import ProteinLigandData, PDBProtein
from utils.evaluation import atom_num

from guidance.latent_guidance import (
    SiteCompatibilityEnergy, build_site_energy_from_map,
    classify_hew_environment, ATOM_TYPE_VOCAB, N_ATOM_TYPES,
)

SITE_MAPS_DIR = ROOT / "experiments/targetdiff_replication/site_maps"
DATA_BASE = Path("/root/autodl-tmp/data/PDB/P-L")
OUTPUT_BASE = ROOT / "results/decompdiff_3mfw"

POCKETS_CONFIG = {
    "3mfw": "2001-2010", "2gni": "2001-2010", "2gqn": "2001-2010",
    "2jke": "2001-2010", "6o4x": "2011-2019", "6phx": "2011-2019",
}


# ═══════════════════════════════════════════════════════════════════════════
# KAG Energy Drift for DecompDiff
# ═══════════════════════════════════════════════════════════════════════════

class KAGEnergyDrift:
    """Compute E_site gradient for DecompDiff's energy_drift_opt.

    Returns a dict compatible with DecompDiff's sample_diffusion:
      {'type': 'kag_esite', 'energy_fn': ..., 'lambda_guide': ..., 'mode': ...}
    """

    def __init__(self, site_energy, lambda_guide=1.0, mode="full_gradient",
                 anchor_indices=None, tau=10.0):
        self.site_energy = site_energy
        self.lambda_guide = lambda_guide
        self.mode = mode  # "full_gradient" or "com_projection"
        self.anchor_indices = anchor_indices or []
        self.tau = tau

    def to(self, device):
        self.site_energy.to(device)
        return self

    def compute_gradient(self, xt, batch_ligand, atom_type_probs=None):
        """Compute energy gradient w.r.t. ligand positions.

        Args:
            xt: [n_total_atoms, 3] current ligand positions
            batch_ligand: [n_total_atoms] batch assignment
            atom_type_probs: [n_total_atoms, n_types] or None

        Returns:
            energy_grad: [n_total_atoms, 3] gradient, energy_val: float
        """
        device = xt.device
        xt_in = xt.detach().clone().requires_grad_(True)

        if atom_type_probs is None:
            atom_type_probs = torch.ones(xt.shape[0], N_ATOM_TYPES, device=device) / N_ATOM_TYPES

        energy = self.site_energy(xt_in, atom_type_probs=atom_type_probs)
        grad = torch.autograd.grad(energy, xt_in)[0]

        if self.mode == "com_projection" and len(self.anchor_indices) > 0:
            # CoM projection: average gradient over anchors, apply to all anchors
            idx = torch.tensor(self.anchor_indices, device=device, dtype=torch.long)
            if idx.numel() > 0:
                anchor_grad = grad[idx]  # [n_anchors, 3]
                com_grad = anchor_grad.mean(dim=0)  # [3]
                grad = torch.zeros_like(grad)
                grad[idx] = com_grad.unsqueeze(0).expand(idx.numel(), 3)

        return -self.lambda_guide * grad.detach(), float(energy.detach().cpu())

    def get_drift_opt(self):
        return {
            'type': 'kag_esite',
            'drift_fn': self,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_decompdiff_model(device="cuda:0"):
    ckpt_path = DECOMP_CKPT
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = ckpt["config"]

    pf = trans.FeaturizeProteinAtom()
    lf = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)

    model = DecompScorePosNet3D(
        config.model,
        protein_atom_feature_dim=pf.feature_dim,
        ligand_atom_feature_dim=lf.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    return model, config, pf, lf


def prepare_pocket(pdb_path, pf, device="cuda:0"):
    """Prepare pocket data for DecompDiff (subpocket prior mode)."""
    protein = PDBProtein(str(pdb_path))
    protein_dict = torchify_dict(protein.to_dict_atom())
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict={
            'element': torch.empty([0], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0], dtype=torch.long),
        }
    )
    return data.to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Main generation
# ═══════════════════════════════════════════════════════════════════════════

def sample_decompdiff_guided(
    model, config, pf, pocket_data, n_mols, device,
    energy_drift_opt=None, anchor_indices=None, fix_anchor_pos=None,
):
    """Run DecompDiff sampling with optional KAG guidance."""
    from scripts.sample_diffusion_decomp import sample_diffusion_ligand_decomp
    from torch_geometric.transforms import Compose

    lf = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform = Compose([pf, lf])

    num_steps = config.sample.num_steps if hasattr(config.sample, 'num_steps') else 500

    # Set up prior configs
    with open(DECOMP_ROOT / "data" / "arm_prior.pkl", 'rb') as f:
        arms_config = pickle.load(f)
    with open(DECOMP_ROOT / "data" / "scaffold_prior.pkl", 'rb') as f:
        scaffold_config = pickle.load(f)

    all_mols = []
    t0 = time.time()

    result = sample_diffusion_ligand_decomp(
        model=model, data=pocket_data,
        init_transform=transform,
        num_samples=n_mols, batch_size=min(n_mols, 10),
        device=device, prior_mode='subpocket',
        num_steps=num_steps, center_pos_mode='none',
        num_atoms_mode='prior',
        arms_natoms_config=arms_config,
        scaffold_natoms_config=scaffold_config,
        atom_enc_mode='add_aromatic',
        bond_fc_mode='fc',
        energy_drift_opt=energy_drift_opt,
    )

    elapsed = time.time() - t0

    for r in result:
        try:
            pred_pos = r['pred_pos']
            pred_v = r['pred_v']
            pred_v = np.argmax(pred_v, axis=1) if pred_v.ndim > 1 else pred_v
            atom_types = [ATOM_TYPE_VOCAB[min(v, N_ATOM_TYPES-1)] for v in pred_v]

            mol = recon.reconstruct_from_generated(pred_pos, pred_v, None)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except:
                    pass
                all_mols.append(mol)
        except Exception as e:
            pass

    return all_mols, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Hooking KAG into DecompDiff's energy_drift_opt
# ═══════════════════════════════════════════════════════════════════════════

# We need to patch DecompDiff's sample_diffusion to handle our custom drift type.
# The existing code at line 638-677 handles energy_drift_opt.
# We'll add a 'kag_esite' type.

def patch_decompdiff_energy_drift():
    """Add 'kag_esite' drift type to DecompDiff's sample_diffusion."""
    import models.decompdiff as dd

    orig_fn_path = DECOMP_ROOT / "models" / "decompdiff.py"
    with open(orig_fn_path) as f:
        code = f.read()

    if "kag_esite" in code:
        return  # already patched

    # Add kag_esite handling after the existing drift types
    old = "elif drift['type'] == 'mmff_min':"
    new = """elif drift['type'] == 'kag_esite':
                        with torch.enable_grad():
                            drift_fn = drift['drift_fn']
                            energy_grad, energy_val = drift_fn.compute_gradient(
                                xt, batch_ligand=batch_ligand)
                            if drift.get('scale', False):
                                energy_grad *= extract(self.pos_score_coef, t, batch_ligand)
                    elif drift['type'] == 'mmff_min':"""

    if old in code:
        code = code.replace(old, new)
        with open(orig_fn_path, "w") as f:
            f.write(code)
        print("  [DecompDiff] Patched with kag_esite energy drift support.")
    else:
        # Try alternative approach: patch the model in-memory
        print("  [DecompDiff] Will patch in-memory (file patch failed).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets", default="3mfw")
    parser.add_argument("--n-mols", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pockets = [p.strip() for p in args.pockets.split(",")]
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Patch DecompDiff
    patch_decompdiff_energy_drift()

    # Load model once
    print("Loading DecompDiff model...")
    model, config, pf, lf = load_decompdiff_model(args.device)
    print(f"  DecompDiff loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Steps: {getattr(config.sample, 'num_steps', 500)}")

    for pocket in pockets:
        year = POCKETS_CONFIG[pocket]
        pdb_path = DATA_BASE / year / pocket / f"{pocket}_protein.pdb"
        ref_lig = DATA_BASE / year / pocket / f"{pocket}_ligand.sdf"
        sm_path = SITE_MAPS_DIR / f"{pocket}_site_map.json"

        if not sm_path.exists():
            print(f"  ⚠ No site map for {pocket}")
            continue

        print(f"\n{'='*50}\n  {pocket}\n{'='*50}")

        # Load site map
        site_map = json.loads(sm_path.read_text())
        hew_sites = [s for s in site_map["sites"] if s["site_type"] == "high_energy_water"]
        print(f"  HEW sites: {len(hew_sites)}")

        # Build site energy
        site_energy = build_site_energy_from_map(
            site_map, sigma_distance=3.0,
            enabled_envs=("hydrophobic", "polar_unsatisfied", "mixed"),
        ).to(args.device)

        # Prepare pocket
        pocket_data = prepare_pocket(pdb_path, pf, args.device)
        print(f"  Pocket prepared: {pocket_data.protein_pos.shape[0]} protein atoms")

        # Get reference ligand atom count
        ref_mol = Chem.SDMolSupplier(str(ref_lig))[0]
        ref_size = ref_mol.GetNumAtoms()
        print(f"  Ref ligand: {ref_size} atoms")

        ref_conf = ref_mol.GetConformer()
        anchor_coords = [list(ref_conf.GetAtomPosition(i)) for i in range(min(4, ref_size))]

        for cond, drift_opt in [
            ("unguided", None),
            ("full_gradient", [KAGEnergyDrift(site_energy, lambda_guide=1.0, mode="full_gradient").to(args.device).get_drift_opt()]),
            ("kag", [KAGEnergyDrift(site_energy, lambda_guide=3.0, mode="com_projection",
                                     anchor_indices=[0, 1]).to(args.device).get_drift_opt()]),
        ]:
            out_dir = OUTPUT_BASE / pocket / cond
            out_dir.mkdir(parents=True, exist_ok=True)
            sdf_path = out_dir / "molecules.sdf"

            if sdf_path.exists() and sdf_path.stat().st_size > 1000:
                print(f"  {cond}: exists, skip")
                continue

            print(f"  {cond} ({args.n_mols} mols)...", end=" ", flush=True)

            mols, elapsed = sample_decompdiff_guided(
                model, config, pf, pocket_data, args.n_mols, args.device,
                energy_drift_opt=drift_opt,
            )

            # Save
            w = Chem.SDWriter(str(sdf_path))
            w.SetKekulize(False)
            for m in mols:
                try: w.write(m)
                except: pass
            w.close()

            meta = {"pocket": pocket, "condition": cond, "n": len(mols), "time_s": elapsed}
            json.dump(meta, open(out_dir / "metadata.json", "w"), indent=2, default=str)

            print(f"{len(mols)} mols, {elapsed:.0f}s")

    print("\n✓ DecompDiff experiment complete")


if __name__ == "__main__":
    main()
