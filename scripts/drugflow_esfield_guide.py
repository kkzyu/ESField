#!/usr/bin/env python3
"""DrugFlow + ESField site-aware energy guidance.

DrugFlow is flow matching — identical architecture to PAFlow.
The ESField gradient is injected via the built-in `guide_log_prob` parameter.

This module is both a CLI script and an importable library for batch drivers.
"""

from __future__ import annotations
import argparse, contextlib, json, os, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from models.potential_network import CompatibilityPotential, CompatibilityPotentialV5, PotentialConfig
from models.site_features import site_type_to_index
from models.analytic_esfield import (
    AnalyticESFieldGuideV2, V6D2Config, create_v6d2_guide,
)

# Default potential checkpoint (v5 if available, else v4)
POTENTIAL_V5_CKPT = f"{ROOT}/experiments/potential_training/v5/potential_v5_epoch_0030.pt"
POTENTIAL_V4_CKPT = f"{ROOT}/experiments/potential_training/train_gpu/compatibility_potential_epoch_0200.pt"
POTENTIAL_DEFAULT_CKPT = POTENTIAL_V5_CKPT if Path(POTENTIAL_V5_CKPT).exists() else POTENTIAL_V4_CKPT

DRUGFLOW_DIR = "/root/baselines/DrugFlow/code/DrugFlow-main"
DRUGFLOW_CKPT = "/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"


# ---------------------------------------------------------------------------
# ESFieldGuide — callable guide_log_prob for DrugFlow's simulate method
# ---------------------------------------------------------------------------

class ESFieldGuide:
    """Callable guide_log_prob for DrugFlow's simulate method.

    Returns site energy. Gradient is taken by DrugFlow's built-in autograd.
    Lower energy = better compatibility, so we return -energy (maximizing log_prob).

    v5 features:
      - d0_override: change the attractive well center (default None = use potential's d0)
      - hew_gating: "all" (default), "nearest" (each atom→nearest HEW only),
                    "top1_conf" (only highest-confidence HEW active)
      - hew_only: if True, only HEW sites contribute to guidance
    """
    SITE_TYPE_MAP = {"unknown": 0, "high_energy_water": 1, "stable_water": 2, "hydrophobic_cavity": 3}

    def __init__(self, potential, site_map, esfield_lambda=1.0, grad_clip=1.0,
                 guidance_start=0.4, guidance_end=0.85,
                 d0_override=None, hew_gating="all", hew_only=True,
                 aggregation="sum"):
        self.potential = potential
        self.esfield_lambda = esfield_lambda
        self.grad_clip = grad_clip
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.d0_override = d0_override
        self.hew_gating = hew_gating
        self.hew_only = hew_only
        self.aggregation = aggregation

        _sm = site_map if isinstance(site_map, dict) else json.loads(Path(site_map).read_text())
        self.sites = _sm["sites"]
        self.site_centers = torch.tensor([s["center"] for s in self.sites], dtype=torch.float32)
        self.site_radii = torch.tensor([s["radius"] for s in self.sites], dtype=torch.float32)
        self.site_confs = torch.tensor([s.get("confidence", 1.0) for s in self.sites], dtype=torch.float32)
        self.site_types = torch.tensor(
            [self.SITE_TYPE_MAP.get(s["site_type"], 0) for s in self.sites], dtype=torch.long)

        # Pre-compute HEW mask
        self.hew_mask = self.site_types == 1  # high_energy_water = 1
        self.n_hew = self.hew_mask.sum().item()

    def to(self, device):
        self.site_centers = self.site_centers.to(device)
        self.site_radii = self.site_radii.to(device)
        self.site_confs = self.site_confs.to(device)
        self.site_types = self.site_types.to(device)
        self.hew_mask = self.hew_mask.to(device)
        return self

    def _apply_gating(self, weight, dist, site_confs):
        """Apply HEW gating to the weight matrix [n_atoms, n_sites]."""
        if self.n_hew == 0:
            return weight

        if self.hew_only:
            # Zero out all non-HEW sites
            weight = weight * self.hew_mask[None, :].float()

        if self.hew_gating == "all":
            pass  # no gating beyond hew_only

        elif self.hew_gating == "nearest":
            # Each atom only responds to its nearest HEW site
            hew_dist = dist.clone()
            hew_dist[:, ~self.hew_mask] = float('inf')
            _, nearest_idx = hew_dist.min(dim=-1)  # [n_atoms]
            gate = torch.zeros_like(weight)
            gate.scatter_(1, nearest_idx.unsqueeze(1), 1.0)
            weight = weight * gate

        elif self.hew_gating == "top1_conf":
            # Only the single highest-confidence HEW site is active
            hew_conf = site_confs * self.hew_mask.float()
            _, top_idx = hew_conf.max(dim=0)  # scalar index
            gate = torch.zeros_like(weight)
            gate[:, top_idx] = 1.0
            weight = weight * gate

        return weight

    def __call__(self, t_array, *, x, h, batch_mask, bonds=None, bond_types=None):
        n_sites = len(self.sites)
        if n_sites == 0:
            return torch.zeros(1, device=x.device)

        # Apply d0 override if set
        if self.d0_override is not None and hasattr(self.potential, 'd0'):
            saved_d0 = self.potential.d0
            self.potential.d0 = self.d0_override
        else:
            saved_d0 = None

        try:
            n_atoms = x.shape[0]
            sites_c = self.site_centers
            rel = x[:, None, :] - sites_c[None, :, :]
            dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)
            sigma = self.site_radii.clamp_min(1e-4)
            weight = torch.exp(-dist ** 2 / (2 * sigma[None, :] ** 2))
            weight = weight * self.site_confs[None, :]

            # Apply HEW gating
            weight = self._apply_gating(weight, dist, self.site_confs)

            atom_probs = F.softmax(h, dim=-1)
            n_atom_types = min(atom_probs.shape[-1], 11)
            nat = n_atoms * n_sites

            rel_tiled = rel.reshape(nat, 3).repeat(n_atom_types, 1)
            dist_tiled = dist.reshape(nat).repeat(n_atom_types)
            st_tiled = self.site_types.repeat(n_atoms).repeat(n_atom_types)
            rad_tiled = self.site_radii.repeat(n_atoms).repeat(n_atom_types)
            conf_tiled = self.site_confs.repeat(n_atoms).repeat(n_atom_types)
            at_all = torch.arange(n_atom_types, device=x.device, dtype=torch.long).repeat_interleave(nat)

            e_all = self.potential(at_all, st_tiled, rel_tiled, dist_tiled, rad_tiled, conf_tiled)
            e_grid = e_all.reshape(n_atom_types, n_atoms, n_sites).permute(1, 2, 0)
            pair_energy = (e_grid * atom_probs[:, :n_atom_types][:, None, :]).sum(dim=-1)
            esfield_energy = (pair_energy * weight).sum()
            if self.aggregation == "sum_norm":
                esfield_energy = esfield_energy / max(n_atoms, 1)

        finally:
            if saved_d0 is not None:
                self.potential.d0 = saved_d0

        return -esfield_energy


# ---------------------------------------------------------------------------
# DrugFlow patching (idempotent — safe to call multiple times)
# ---------------------------------------------------------------------------

def _patch_drugflow_lightning():
    """Patch DrugFlow's lightning.py to enable ESField guidance. Idempotent."""
    import shutil
    lmod_path = os.path.join(DRUGFLOW_DIR, "src/model/lightning.py")
    backup = lmod_path + ".bak"
    if not os.path.exists(backup):
        shutil.copy(lmod_path, backup)

    with open(lmod_path) as f:
        code = f.read()

    # Check if already patched
    if "hasattr(guide_log_prob, 'guidance_start')" in code:
        return  # already patched

    old = """            if guide_log_prob is not None:
                raise NotImplementedError('Not yet implemented for flow matching model')
                alpha_t = self.diffusion_x.schedule.alpha(self.gamma_x(t_array))

                with torch.enable_grad():
                    zt_x_ligand.requires_grad = True
                    g = guide_log_prob(t_array, x=ligand['x'], h=ligand['h'], batch_mask=ligand['mask'],
                                       bonds=ligand['bonds'], bond_types=ligand['e'])

                    # Compute gradient w.r.t. coordinates
                    grad_x_lig = torch.autograd.grad(g.sum(), inputs=ligand['x'])[0]

                    # clip gradients
                    g_max = 1.0
                    clip_mask = (grad_x_lig.norm(dim=-1) > g_max)
                    grad_x_lig[clip_mask] = \\
                        grad_x_lig[clip_mask] / grad_x_lig[clip_mask].norm(
                            dim=-1, keepdim=True) * g_max

                delta_eps_lig = -1 * (1 - alpha_t[lig_mask]).sqrt() * grad_x_lig
            else:
                delta_eps_lig = None"""

    new = """            delta_eps_lig = None
            if guide_log_prob is not None and hasattr(guide_log_prob, 'guidance_start'):
                gs, ge = guide_log_prob.guidance_start, guide_log_prob.guidance_end
                lam = guide_log_prob.esfield_lambda
                lam_t = lam if (gs <= float(t) <= ge) else 0.0
                if lam_t > 0:
                    with torch.enable_grad():
                        ligand_tmp = ligand['x'].detach().requires_grad_(True)
                        g = guide_log_prob(t_array, x=ligand_tmp, h=ligand['h'],
                                           batch_mask=ligand['mask'],
                                           bonds=ligand.get('bonds'),
                                           bond_types=ligand.get('e'))
                        grad_x = torch.autograd.grad(g.sum(), ligand_tmp)[0]
                        gnorm = grad_x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        grad_x = grad_x * torch.clamp(guide_log_prob.grad_clip / gnorm, max=1.0)
                    delta_eps_lig = lam_t * grad_x"""

    if old not in code:
        raise RuntimeError(
            "DrugFlow lightning.py has unexpected content — cannot patch. "
            "Check if DrugFlow source code has changed.")
    code = code.replace(old, new)
    with open(lmod_path, "w") as f:
        f.write(code)


# ---------------------------------------------------------------------------
# DrugFlow model loading context
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _drugflow_import_context():
    """Context manager for DrugFlow imports (cwd + sys.path)."""
    saved_cwd = os.getcwd()
    os.chdir(DRUGFLOW_DIR)
    # Use absolute paths to avoid conflicts with ESField's src package
    drugflow_src = os.path.join(DRUGFLOW_DIR, "src")
    sys.path.insert(0, drugflow_src)
    sys.path.insert(0, DRUGFLOW_DIR)
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        try:
            sys.path.remove(DRUGFLOW_DIR)
        except ValueError:
            pass
        try:
            sys.path.remove(drugflow_src)
        except ValueError:
            pass


def load_esfield_potential(checkpoint_path, device="cuda:0"):
    """Load CompatibilityPotential (v4 or v5) from checkpoint. Auto-detects version."""
    pot_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pot_cfg = pot_ckpt["config"]
    is_v5 = "auc_distance_matched" in pot_ckpt

    if is_v5:
        model = CompatibilityPotentialV5(PotentialConfig(
            atom_embed_dim=pot_cfg["atom_embed_dim"],
            site_embed_dim=pot_cfg["site_embed_dim"],
            hidden_dim=pot_cfg["hidden_dim"],
            num_layers=pot_cfg["num_layers"],
        )).to(device).eval()
        print(f"  Loaded Potential v5 (hand-crafted energy shape)")
    else:
        model = CompatibilityPotential(PotentialConfig(
            atom_embed_dim=pot_cfg["atom_embed_dim"],
            site_embed_dim=pot_cfg["site_embed_dim"],
            hidden_dim=pot_cfg["hidden_dim"],
            num_layers=pot_cfg["num_layers"],
        )).to(device).eval()
        print(f"  Loaded Potential v4 (pure MLP energy)")
    model.load_state_dict(pot_ckpt["model_state_dict"])
    return model


def load_drugflow_model(checkpoint_path=DRUGFLOW_CKPT, device="cuda:0"):
    """Load DrugFlow model with ESField guidance patch applied.

    Patches DrugFlow's lightning.py on first call (idempotent).
    Returns the loaded model.
    """
    warnings.filterwarnings("ignore")

    # Temporarily override torch.load for the DrugFlow checkpoint
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        _patch_drugflow_lightning()
        with _drugflow_import_context():
            from src.model import lightning as lmod
            model = lmod.DrugFlow.load_from_checkpoint(checkpoint_path, map_location=device)
    finally:
        torch.load = _orig_load

    model = model.to(device).eval()
    return model


def process_protein_for_drugflow(protein_pdb, ref_ligand, model):
    """Process protein+ligand into DrugFlow's batch format.

    Returns: (data_dict, mol_size) where data_dict is on CPU.
    """
    from Bio.PDB import PDBParser

    pdb_model = PDBParser(QUIET=True).get_structure("", protein_pdb)[0]
    rdmol = Chem.SDMolSupplier(ref_ligand)[0]
    ref_size = rdmol.GetNumAtoms()

    with _drugflow_import_context():
        from src.data.data_utils import process_raw_pair, TensorDict
        from src.data.dataset import ProcessedLigandPocketDataset
        from torch.utils.data import DataLoader
        from functools import partial

        ligand_raw, pocket_raw = process_raw_pair(
            pdb_model, rdmol, dist_cutoff=8.0,
            pocket_representation=model.pocket_representation,
            compute_nerf_params=True)
        ligand_raw["name"] = "ligand"

        collate = partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None)
        data = next(iter(DataLoader(
            [{"ligand": ligand_raw, "pocket": pocket_raw}],
            batch_size=1, collate_fn=collate)))
    return data, ref_size


def generate_molecules(model, potential, protein_pdb, ref_ligand, site_map_path,
                       output_dir, *, num_samples=20, mol_size=None,
                       esfield_lambda=1.0, device="cuda:0", timesteps=40,
                       seed=42, gen_batch_size=5, guidance_start=0.4,
                       guidance_end=0.85, save_sdf=True,
                       d0_override=None, hew_gating="all",
                       aggregation="sum",
                       guide=None,
                       guide_type="learned_v5",
                       v6_config=None):
    """Generate molecules with ESField guidance.

    Args:
        model: loaded DrugFlow model
        potential: loaded CompatibilityPotential (unused if guide_type="analytic_v6")
        protein_pdb, ref_ligand, site_map_path: paths to input files
        output_dir: directory for output SDF and metadata
        num_samples: number of molecules to generate
        mol_size: number of heavy atoms (auto-detect from ref if None)
        esfield_lambda: guidance strength (0 = no guidance)
        device: torch device
        timesteps: ODE integration steps
        seed: random seed
        gen_batch_size: molecules per GPU batch
        guidance_start, guidance_end: guidance schedule bounds
        save_sdf: whether to write SDF file
        guide: pre-built guide object (overrides potential + guide_type if provided)
        guide_type: "learned_v5" (default) or "analytic_v6"
        v6_config: dict of V6DConfig overrides (only used with guide_type="analytic_v6")

    Returns:
        dict with keys: sdf_path, sampled_mols, valid_count, elapsed, metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build guide
    if guide is not None:
        # Use pre-built guide (already on device)
        pass
    elif guide_type in ("analytic_v6", "analytic_v6d2") :
        v6d2_kwargs = {
            "esfield_lambda": esfield_lambda,
            "guidance_start": guidance_start,
            "guidance_end": guidance_end,
        }
        if v6_config:
            v6d2_kwargs.update(v6_config)
        guide = create_v6d2_guide(site_map_path, **v6d2_kwargs).to(device)
        potential = None  # not used
    else:
        guide = ESFieldGuide(
            potential, site_map_path, esfield_lambda=esfield_lambda,
            guidance_start=guidance_start, guidance_end=guidance_end,
            d0_override=d0_override, hew_gating=hew_gating,
            aggregation=aggregation).to(device)

    # Process protein
    data, ref_size = process_protein_for_drugflow(protein_pdb, ref_ligand, model)
    mol_size = mol_size or ref_size

    # Move data to device
    from src.data.data_utils import TensorDict
    new_data = {
        "ligand": TensorDict(**data["ligand"]).to(device),
        "pocket": TensorDict(**data["pocket"]).to(device),
    }

    # Generate in batches
    guide_fn = guide if (esfield_lambda > 0 and guide is not None) else None
    t0 = time.time()
    sampled = []
    from tqdm import tqdm
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, gen_batch_size),
                                desc=f"  Gen λ={esfield_lambda}", leave=False):
            n_this_batch = min(gen_batch_size, num_samples - batch_start)
            rdmols, _, _ = model.sample(
                new_data, n_samples=n_this_batch, timesteps=timesteps,
                num_nodes=mol_size, guide_log_prob=guide_fn)
            sampled.extend(rdmols)
    elapsed = time.time() - t0

    # Write SDF
    sdf_path = str(output_dir / "molecules.sdf")
    if save_sdf:
        writer = Chem.SDWriter(sdf_path)
        writer.SetKekulize(False)
        valid_mols = [m for m in sampled if m is not None]
        for m in valid_mols:
            try:
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass
            writer.write(m)
        writer.close()

    # Compute basic metrics
    mols = list(Chem.SDMolSupplier(sdf_path, sanitize=False)) if save_sdf else []
    from rdkit.Chem import QED, Descriptors
    qeds, mws, logps = [], [], []
    for m in mols:
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            qeds.append(QED.qed(m))
            mws.append(Descriptors.MolWt(m))
            logps.append(Descriptors.MolLogP(m))
        except Exception:
            pass

    metrics = {
        "valid": len([m for m in sampled if m is not None]),
        "total": num_samples,
        "qed_mean": float(np.mean(qeds)) if qeds else 0.0,
        "qed_std": float(np.std(qeds)) if qeds else 0.0,
        "mw_mean": float(np.mean(mws)) if mws else 0.0,
        "logp_mean": float(np.mean(logps)) if logps else 0.0,
        "time_per_sample": elapsed / num_samples,
        "total_time": elapsed,
    }

    # Save metadata
    meta = {
        "protein_pdb": str(protein_pdb),
        "ref_ligand": str(ref_ligand),
        "site_map": str(site_map_path),
        "esfield_lambda": esfield_lambda,
        "timesteps": timesteps,
        "num_samples": num_samples,
        "mol_size": mol_size,
        "seed": seed,
        "metrics": metrics,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {
        "sdf_path": sdf_path,
        "sampled_mols": sampled,
        "valid_count": metrics["valid"],
        "elapsed": elapsed,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DrugFlow + ESField guided generation")
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--ref-ligand", required=True)
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--potential-ckpt",
                        default=POTENTIAL_DEFAULT_CKPT)
    parser.add_argument("--drugflow-ckpt", default=DRUGFLOW_CKPT)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--mol-size", type=int, default=None)
    parser.add_argument("--esfield-lambda", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-batch-size", type=int, default=5)
    parser.add_argument("--guidance-start", type=float, default=0.4)
    parser.add_argument("--guidance-end", type=float, default=0.85)
    parser.add_argument("--guide-type", default="learned_v5",
                        choices=["learned_v5", "analytic_v6", "analytic_v6d2"],
                        help="learned_v5: MLP alpha/beta; analytic_v6: v6-D rule-based; analytic_v6d2: v6-D.2 capture+occupancy")
    # v6-D specific options
    parser.add_argument("--v6-sigma-occ", type=float, default=1.2,
                        help="v6-D: Gaussian occupancy width")
    parser.add_argument("--v6-disp-weight", type=float, default=1.0,
                        help="v6-D: displacement reward weight")
    parser.add_argument("--v6-wrong-atom-weight", type=float, default=0.5,
                        help="v6-D: wrong atom penalty weight")
    parser.add_argument("--v6-clash-weight", type=float, default=1.0,
                        help="v6-D: protein clash penalty weight")
    parser.add_argument("--v6-overfill-weight", type=float, default=0.3,
                        help="v6-D: overfill penalty weight")
    parser.add_argument("--v6-min-confidence", type=float, default=0.3,
                        help="v6-D: minimum HEW site confidence")
    parser.add_argument("--v6-top-k", type=int, default=0,
                        help="v6-D: top-k HEW selection (0=all)")
    parser.add_argument("--v6-filter-mixed", action="store_true",
                        help="v6-D: filter low-confidence mixed HEW sites")
    parser.add_argument("--v6-cutoff-dist", type=float, default=5.0,
                        help="v6-D: interaction cutoff distance")
    # v5 specific options
    parser.add_argument("--v5-d0-override", type=float, default=None,
                        help="v5: override attractive well center distance")
    parser.add_argument("--v5-hew-gating", default="all",
                        choices=["all", "nearest", "top1_conf"],
                        help="v5: HEW gating strategy")
    parser.add_argument("--v5-aggregation", default="sum",
                        choices=["sum", "sum_norm"],
                        help="v5: energy aggregation method")
    args = parser.parse_args()

    device = args.device

    if args.guide_type in ("analytic_v6", "analytic_v6d2"):
        print(f"Guide type: {args.guide_type}")
        potential = None  # analytic guides don't need a learned potential
    else:
        print(f"Loading potential from {args.potential_ckpt}")
        potential = load_esfield_potential(args.potential_ckpt, device=device)

    print(f"Loading DrugFlow from {args.drugflow_ckpt}")
    model = load_drugflow_model(args.drugflow_ckpt, device=device)
    print(f"DrugFlow: {sum(p.numel() for p in model.parameters()):,} params")

    # Build v6-D config dict if needed
    v6_config = None
    if args.guide_type in ("analytic_v6", "analytic_v6d2"):
        v6_config = {
            "sigma_occ": args.v6_sigma_occ,
            "sigma_cap": 2.5,
            "disp_weight": args.v6_disp_weight,
            "wrong_atom_weight": args.v6_wrong_atom_weight,
            "clash_weight": args.v6_clash_weight,
            "overfill_weight": args.v6_overfill_weight,
            "min_confidence": args.v6_min_confidence,
            "top_k": args.v6_top_k,
            "cutoff_dist": args.v6_cutoff_dist,
            "enabled_envs": ("hydrophobic",),
        }

    result = generate_molecules(
        model, potential,
        protein_pdb=args.protein_pdb,
        ref_ligand=args.ref_ligand,
        site_map_path=args.site_map,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        mol_size=args.mol_size,
        esfield_lambda=args.esfield_lambda,
        device=device,
        timesteps=args.timesteps,
        seed=args.seed,
        gen_batch_size=args.gen_batch_size,
        guidance_start=args.guidance_start,
        guidance_end=args.guidance_end,
        guide_type=args.guide_type,
        d0_override=args.v5_d0_override,
        hew_gating=args.v5_hew_gating,
        aggregation=args.v5_aggregation,
        v6_config=v6_config,
    )

    m = result["metrics"]
    print(f"\n=== Results ===")
    print(f"Valid: {m['valid']}/{m['total']}")
    if m["qed_mean"]:
        print(f"QED:  {m['qed_mean']:.3f} ± {m['qed_std']:.3f}")
        print(f"MW:   {m['mw_mean']:.0f}")
        print(f"logP: {m['logp_mean']:.1f}")
    print(f"Time:  {result['elapsed']:.1f}s ({m['time_per_sample']:.1f}s/sample)")
    print(f"Saved: {result['sdf_path']}")


if __name__ == "__main__":
    main()
