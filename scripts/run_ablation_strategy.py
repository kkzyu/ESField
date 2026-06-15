#!/usr/bin/env python3
"""
Task 4: Orthogonal Decomposition Ablation — Real DrugFlow Integration.

Three strategies for decomposing the site-compatibility gradient ∇E_site:
  1. full_gradient:       Per-atom gradient (R^{3N}), naive global injection
  2. internal_projection: Internal component only (ΣΔx_int = 0), CoM fixed
  3. com_projection:      CoM only (Ours), zero strain, Theorem 1 guarantee

Uses real DrugFlow ODE sampling loop with the post_step_callback hook.
"""

from __future__ import annotations

import argparse, json, os, sys, time, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASELINES = Path("/root/baselines")
DATA_ROOT = Path("/root/autodl-tmp/data")
DRUGFLOW_DIR = BASELINES / "DrugFlow/code/DrugFlow-main"
DRUGFLOW_CKPT = Path("/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt")

# CRITICAL import order:
# 1. DrugFlow root first so `from src.xxx` resolves to DrugFlow's src/
# 2. DrugFlow src/ so direct subpackage imports work
# 3. ESField src/ so `from guidance.xxx` works (NOT `from src.xxx`)
# 4. NEVER add ROOT to path — it would shadow DrugFlow's src/
sys.path.insert(0, str(DRUGFLOW_DIR))
sys.path.insert(0, str(DRUGFLOW_DIR / "src"))
sys.path.insert(0, str(ROOT / "src"))
# ROOT is NOT added — prevents ESField's src/ from shadowing DrugFlow
# For scripts.* imports, use explicit path:
sys.path.insert(0, str(ROOT / "scripts"))

from Bio.PDB import PDBParser
from rdkit import Chem
from torch.utils.data import DataLoader
from functools import partial

from src.data.dataset import ProcessedLigandPocketDataset
from src.data.data_utils import TensorDict, process_raw_pair
from src.model.lightning import DrugFlow
from src import utils as drugflow_utils

from guidance.latent_guidance import SiteCompatibilityEnergy
from guidance.kinematic_anchor import KinematicScheduler
from kpe_instrumentation import KPETracker, KPELogger

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Pocket config
# ═══════════════════════════════════════════════════════════════════════════

POCKET_CONFIG = {
    "3mfw": {"year": "2001-2010", "ref_atoms": 26, "n_hew": 7},
    "2gni": {"year": "2001-2010", "ref_atoms": 20, "n_hew": 3},
    "6o4x": {"year": "2011-2019", "ref_atoms": 22, "n_hew": 6},
    "2jke": {"year": "2001-2010", "ref_atoms": 24, "n_hew": 4},
    "2gqn": {"year": "2001-2010", "ref_atoms": 18, "n_hew": 7},
    "6phx": {"year": "2011-2019", "ref_atoms": 21, "n_hew": 5},
}


def get_pocket_paths(pocket_id: str) -> dict:
    cfg = POCKET_CONFIG[pocket_id]
    year = cfg["year"]
    base = DATA_ROOT / "PDB/P-L" / year / pocket_id
    protein_pdb = base / f"{pocket_id}_protein.pdb"
    ref_ligand = base / f"{pocket_id}_ligand.sdf"
    site_map = ROOT / "experiments/targetdiff_replication/site_maps" / f"{pocket_id}_site_map.json"
    return {
        "protein_pdb": protein_pdb, "ref_ligand": ref_ligand,
        "site_map": site_map if site_map.exists() else None,
        "ref_atoms": cfg["ref_atoms"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# E_site gradient
# ═══════════════════════════════════════════════════════════════════════════

def _classify_microenvironment(features: dict) -> int:
    """Map site features to microenvironment index.
    0=hydrophobic, 1=polar_unsatisfied, 2=mixed, 3=buried
    """
    hb = features.get("hbond_count", 0)
    hc = features.get("hydrophobic_contact_count", 0)
    # Simple rule matching paper's classification
    if hc >= 3 and hb <= 1:
        return 0  # hydrophobic-dominated
    elif hb <= 1 and hc < 3:
        return 1  # polar-unsatisfied
    else:
        return 2  # mixed


def build_site_energy(site_map_path) -> SiteCompatibilityEnergy | None:
    if site_map_path is None or not Path(site_map_path).exists():
        return None
    with open(site_map_path) as f:
        site_data = json.load(f)
    # Site map format: {"sites": [...], ...}
    if isinstance(site_data, dict):
        sites = site_data.get("sites", [])
    else:
        sites = site_data if isinstance(site_data, list) else []

    hew_sites = [s for s in sites if s.get("site_type") == "high_energy_water"]
    if not hew_sites:
        print(f"  WARNING: 0 HEW sites found in {site_map_path}")
        return None

    centers = torch.tensor([s["center"] for s in hew_sites], dtype=torch.float32)
    env_indices = torch.tensor(
        [_classify_microenvironment(s.get("features", {})) for s in hew_sites],
        dtype=torch.long)
    confidences = torch.tensor(
        [s.get("confidence", 1.0) for s in hew_sites], dtype=torch.float32)

    se = SiteCompatibilityEnergy()
    se.register_sites(centers, env_indices, confidences)
    return se


def compute_esite_gradient_analytic(x, site_energy, sigma=3.0):
    """Per-atom ∇E_site via analytic Gaussian derivative."""
    n_atoms, device = x.shape[0], x.device
    sigma2 = 2.0 * sigma ** 2
    centers = site_energy._site_centers.to(device)
    env_indices = site_energy._site_env_indices.to(device)
    compat = site_energy.compatibility_matrix.to(device)
    K = centers.shape[0]
    grad = torch.zeros(n_atoms, 3, device=device)
    for k in range(K):
        c_k = centers[k]
        env_k = env_indices[k].item()
        compat_k = compat[env_k]
        rel = x - c_k.unsqueeze(0)
        dist_sq = (rel ** 2).sum(dim=-1)
        gauss = torch.exp(-dist_sq / sigma2)
        best_compat = compat_k.max()
        contrib = best_compat * gauss.unsqueeze(-1) * rel / sigma2
        grad -= contrib / max(K, 1)
    return grad


# ═══════════════════════════════════════════════════════════════════════════
# Ablation callback (DrugFlow post_step_callback interface)
# ═══════════════════════════════════════════════════════════════════════════

class AblationCallback:
    def __init__(self, strategy, site_energy, total_steps=100, lambda_max=1.0,
                 profile="quadratic", grad_clip=0.5, device="cuda",
                 kpe_tracker=None):
        self.strategy = strategy
        self.site_energy = site_energy
        self.total_steps = total_steps
        self.grad_clip = grad_clip
        self.device = device
        self.kpe_tracker = kpe_tracker
        self.scheduler = KinematicScheduler(lambda_max=lambda_max, profile=profile)
        self._call_count = 0
        self._x_prev = None
        self._grad_norms = []
        self.dt = 1.0 / max(total_steps, 1)

    def __call__(self, ligand, step_idx, t_val):
        self._call_count += 1
        x = ligand["x"]

        # baseline: no guidance — pure DrugFlow ODE
        if self.strategy == "baseline":
            if self._x_prev is None: self._x_prev = x.clone()
            else: self._x_prev = x.clone()
            return ligand

        # hard_fix: overwrite first 4 anchor atoms to nearest HEW site
        if self.strategy == "hard_fix":
            if self._x_prev is None: self._x_prev = x.clone(); return ligand
            if self.site_energy is None or self.site_energy._site_centers is None:
                self._x_prev = x.clone(); return ligand
            if t_val < 0.6: self._x_prev = x.clone(); return ligand

            centers = self.site_energy._site_centers.to(x.device)
            n_anchors = min(4, x.shape[0])
            # For each anchor atom i, snap to nearest HEW site
            for i in range(n_anchors):
                dists = torch.norm(x[i] - centers, dim=-1)
                nearest = torch.argmin(dists)
                x[i] = centers[nearest]
            ligand["x"] = x

            if self.kpe_tracker is not None:
                v_eff = x - self._x_prev
                v_guide = x - self._x_prev  # hard-fix: all displacement is "guidance"
                self.kpe_tracker.record_step(step_idx, t_val, v_eff, v_guide)
            self._x_prev = x.clone()
            return ligand

        # Gradient-based strategies (full_gradient, internal_projection, com_projection):
        # Phase gate: only guide during Phase 2 (geometric refinement, t >= 0.6).
        if t_val < 0.6:
            if self._x_prev is None:
                self._x_prev = ligand["x"].clone()
            else:
                self._x_prev = ligand["x"].clone()
            return ligand

        x = ligand["x"]
        n_atoms = x.shape[0]
        if n_atoms < 2 or self.site_energy is None:
            return ligand

        if self._x_prev is None:
            self._x_prev = x.clone()
            return ligand

        grad = compute_esite_gradient_analytic(x, self.site_energy)
        if grad.norm() < 1e-8:
            self._x_prev = x.clone()
            return ligand

        lam = self.scheduler(t_val)
        if isinstance(lam, torch.Tensor):
            lam = lam.item()

        if self.strategy == "full_gradient":
            correction = lam * grad
            grad_guide = grad
        elif self.strategy == "internal_projection":
            grad_com = grad.mean(dim=0, keepdim=True)
            grad_int = grad - grad_com
            correction = lam * grad_int
            grad_guide = grad_int
        elif self.strategy == "com_projection":
            grad_com = grad.mean(dim=0, keepdim=True)
            correction = lam * grad_com.expand_as(grad)
            grad_guide = grad_com.expand_as(grad)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        corr_norm = correction.norm(dim=-1).max().item()
        if corr_norm > self.grad_clip:
            correction = correction * (self.grad_clip / max(corr_norm, 1e-8))

        # CRITICAL KPE fix: save post-ODE x BEFORE applying guidance.
        # v_eff = x_ode_current - x_ode_prev = pure ODE displacement.
        # v_guide = correction = guidance displacement (NOT mixing prev guidance).
        x_ode = x.clone()  # post-ODE, pre-guidance
        ligand["x"] = x + correction  # apply guidance

        if self.kpe_tracker is not None:
            v_eff_ode = x_ode - self._x_prev  # pure ODE displacement
            v_guide = correction               # this step's guidance only
            self.kpe_tracker.record_step(step_idx, t_val, v_eff_ode, v_guide)

        self._grad_norms.append(float(grad.norm().item()))
        self._x_prev = x_ode  # save post-ODE (NOT post-guidance) for next v_eff
        return ligand


# ═══════════════════════════════════════════════════════════════════════════
# DrugFlow generation
# ═══════════════════════════════════════════════════════════════════════════

def run_drugflow_ablation(pocket_id, strategy, n_samples=50, n_steps=100,
                          lambda_max=1.0, output_dir=".", device="cuda",
                          batch_size=8, seed=42, protein_pdb=None, site_json=None):
    drugflow_utils.set_deterministic(seed=seed)
    drugflow_utils.disable_rdkit_logging()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = out_dir / "sdfs"
    sdf_dir.mkdir(exist_ok=True)

    paths = get_pocket_paths(pocket_id)
    # Allow CLI overrides for protein and site paths
    _protein_pdb = Path(protein_pdb) if protein_pdb else paths["protein_pdb"]
    _ref_ligand = paths["ref_ligand"]
    _site_map = Path(site_json) if site_json else paths.get("site_map")
    ref_atoms = paths.get("ref_atoms", 25)

    if not _protein_pdb.exists():
        raise FileNotFoundError(f"Protein not found: {_protein_pdb}")
    if not _ref_ligand.exists():
        raise FileNotFoundError(f"Ligand not found: {_ref_ligand}")

    print(f"  Protein: {_protein_pdb}")
    print(f"  Ref ligand: {_ref_ligand} ({ref_atoms} atoms)")

    site_energy = build_site_energy(_site_map)
    if site_energy is None:
        print(f"  WARNING: No HEW sites — using zero-guidance fallback")
        # Return early with empty results
        return {"status": "skipped", "reason": "no_site_map", "n_molecules": 0}

    print(f"  HEW sites: {site_energy.n_sites}")

    print(f"  Loading DrugFlow...")
    t0 = time.time()
    # PyTorch 2.6+ Lightning passes weights_only=True to torch.load.
    # DrugFlow ckpt serializes argparse.Namespace, pathlib paths, OrderedDict.
    import argparse as _ap, pathlib as _pl, collections as _col
    torch.serialization.add_safe_globals([
        _ap.Namespace,
        _pl.PosixPath, _pl.WindowsPath, _pl.PurePosixPath, _pl.PureWindowsPath,
        _col.OrderedDict,
    ])
    model = DrugFlow.load_from_checkpoint(str(DRUGFLOW_CKPT), map_location=device, strict=False)
    model.datadir = str(DRUGFLOW_DIR / "src" / "default")
    model.setup(stage="generation")
    model.batch_size = model.eval_batch_size = batch_size
    model.eval().to(device)
    if n_steps is not None:
        model.T = n_steps
    print(f"  Model ready in {time.time()-t0:.1f}s, T={model.T}")

    pdb_parser = PDBParser(QUIET=True)
    pdb_model = pdb_parser.get_structure("", str(_protein_pdb))[0]
    rdmol = Chem.SDMolSupplier(str(_ref_ligand))[0]
    if rdmol is None:
        mol2 = str(_ref_ligand).replace(".sdf", ".mol2")
        rdmol = Chem.MolFromMol2File(mol2)
    if rdmol is None:
        raise RuntimeError(f"Cannot read ligand: {_ref_ligand}")

    ligand, pocket = process_raw_pair(
        pdb_model, rdmol, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=str(_protein_pdb) if model.dynamics.add_nma_feat else None)
    ligand["name"] = "ligand"

    kpe_logger = KPELogger(condition_name=strategy, pocket_name=pocket_id,
                           output_dir=str(out_dir / "kpe"))
    n_generated = 0
    molecule_size = ref_atoms

    print(f"  Generating {n_samples} molecules (strategy={strategy})...")
    with torch.no_grad():
        while n_generated < n_samples:
            dataset = [{"ligand": ligand, "pocket": pocket} for _ in range(batch_size)]
            dataloader = DataLoader(
                dataset=dataset, batch_size=batch_size,
                collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
                pin_memory=True)

            for data in dataloader:
                new_data = {
                    "ligand": TensorDict(**data["ligand"]).to(device),
                    "pocket": TensorDict(**data["pocket"]).to(device),
                }
                n_at = new_data["ligand"]["x"].shape[1]
                kpe_t = kpe_logger.new_molecule(n_atoms=n_at, total_steps=n_steps, framework="ode")
                cb = AblationCallback(strategy=strategy, site_energy=site_energy,
                                      total_steps=n_steps, lambda_max=lambda_max,
                                      device=device, kpe_tracker=kpe_t)

                rdmols, rdpockets, _ = model.sample(
                    new_data, n_samples=1, timesteps=n_steps,
                    num_nodes=molecule_size, post_step_callback=cb)
                kpe_logger.finish_molecule()

                for mol in rdmols:
                    if mol is not None and n_generated < n_samples:
                        sdf_path = sdf_dir / f"mol_{n_generated:03d}.sdf"
                        w = Chem.SDWriter(str(sdf_path))
                        try:
                            w.write(mol)
                        except Exception:
                            pass
                        w.close()
                        n_generated += 1
                if n_generated >= n_samples:
                    break

    elapsed = time.time() - t0
    print(f"  Done: {n_generated} mols in {elapsed:.1f}s ({elapsed/max(n_generated,1):.1f}s/mol)")

    kpe_logger.save()

    meta = {"pocket": pocket_id, "strategy": strategy, "n_samples": n_generated,
            "n_steps": n_steps, "lambda_max": lambda_max, "ref_atoms": ref_atoms,
            "elapsed_s": elapsed}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {"status": "completed", "n_molecules": n_generated, **meta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket", required=True)
    parser.add_argument("--strategy", required=True,
                       choices=["full_gradient", "internal_projection", "com_projection",
                                "baseline", "hard_fix"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--protein-pdb", default=None, help="Override protein PDB path")
    parser.add_argument("--site-json", default=None, help="Override site map JSON path")
    args = parser.parse_args()

    result = run_drugflow_ablation(
        pocket_id=args.pocket, strategy=args.strategy,
        n_samples=args.n_samples, n_steps=args.n_steps,
        lambda_max=args.lambda_max, output_dir=args.output_dir,
        device=args.device, batch_size=args.batch_size,
        protein_pdb=args.protein_pdb, site_json=args.site_json)
    print(json.dumps(result, indent=2, default=str))
