#!/usr/bin/env python3
"""TargetDiff + ESField guided generation adapter.
Injects site energy gradient into TargetDiff's DDPM x0 prediction before posterior.

Usage:
  python scripts/targetdiff_esfield_guide.py \
    --protein-pdb /path/to/pocket.pdb \
    --site-map experiments/potential_training/site_maps/XXX_site_map.json \
    --output-dir experiments/targetdiff/guided \
    --num-samples 5
"""

from __future__ import annotations
import argparse, json, os, sys, time, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TD = Path("/root/baselines/TargetDiff/code/targetdiff-main")

# Import ESField first (before TD shadows models/), save refs
sys.path.insert(0, str(ROOT / "src"))
from models.potential_network import CompatibilityPotential, PotentialConfig  # noqa
# Then put TD first for its imports
sys.path.insert(0, str(TD))

import numpy as np
import torch
from torch_scatter import scatter_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--site-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--potential-ckpt",
                        default=f"{ROOT}/experiments/potential_training/train_gpu/compatibility_potential_epoch_0200.pt")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esfield-lambda", type=float, default=0.5)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear ESField-cached modules so TD imports resolve correctly
    for k in list(sys.modules):
        if k.startswith(('utils', 'models', 'datasets')):
            del sys.modules[k]

    # --- Load everything ---
    from models.molopt_score_model import ScorePosNet3D
    from utils import transforms as trans
    from datasets.pl_data import FOLLOW_BATCH
    from utils.evaluation import atom_num
    from utils import reconstruct
    from scripts.sample_for_pocket import pdb_to_pocket_data
    from torch_geometric.data import Batch
    from torch_geometric.transforms import Compose

    ckpt = torch.load("pretrained_models/pretrained_diffusion.pt", map_location=args.device, weights_only=False)
    config = ckpt["config"]
    pf = trans.FeaturizeProteinAtom()
    lf = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    model = ScorePosNet3D(config.model, pf.feature_dim, lf.feature_dim).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"TargetDiff: {sum(p.numel() for p in model.parameters()):,} params")

    with open(args.site_map) as f: site_map = json.load(f)
    print(f"Site map: {len(site_map['sites'])} sites")

    pot_ckpt = torch.load(args.potential_ckpt, map_location="cpu", weights_only=False)
    pot_cfg = pot_ckpt.get("config", {})
    potential = CompatibilityPotential(PotentialConfig(
        atom_embed_dim=pot_cfg["atom_embed_dim"],
        site_embed_dim=pot_cfg["site_embed_dim"],
        hidden_dim=pot_cfg["hidden_dim"],
        num_layers=pot_cfg["num_layers"],
    )).to(args.device).eval()
    potential.load_state_dict(pot_ckpt["model_state_dict"])
    print(f"Potential: epoch {pot_ckpt.get('epoch','?')}")

    # Prepare pocket
    data = Compose([pf])(pdb_to_pocket_data(args.protein_pdb))
    print(f"Protein: {len(data.protein_pos)} atoms")

    if args.dry_run:
        print("[DRY RUN] Not generating")
        return

    # --- Guided sampling ---
    print(f"\nGenerating {args.num_samples} molecules (lambda={args.esfield_lambda})...")
    t0 = time.time()

    class Guidance:
        def __init__(self, pot): self.potential = pot
    g = Guidance(potential) if potential else None

    all_pos, all_v = [], []
    for si in range(args.num_samples):
        batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(args.device)
        bp = batch.protein_element_batch
        center_pos_b = scatter_mean(batch.protein_pos, bp, dim=0)
        psize = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
        n_at = int(atom_num.sample_atom_num(psize).item())
        bl = torch.tensor([0]*n_at).to(args.device)
        ipos = center_pos_b[bl] + torch.randn_like(center_pos_b[bl])
        iv = torch.randint(0, model.num_classes, (n_at,)).to(args.device)

        r = model.sample_diffusion_guided(
            protein_pos=batch.protein_pos, protein_v=batch.protein_atom_feature.float(),
            batch_protein=bp, init_ligand_pos=ipos, init_ligand_v=iv, batch_ligand=bl,
            num_steps=args.num_steps, pos_only=False, center_pos_mode="protein",
            site_guidance=g, site_map=site_map,
            esfield_lambda_max=args.esfield_lambda,
        )
        all_pos.append(r["pos"].cpu())
        all_v.append(r["v"].cpu())
        del batch, bp, bl, ipos, iv, r
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/args.num_samples:.1f}s/sample)")

    combined = {"pos": torch.cat(all_pos, dim=0), "v": torch.cat(all_v, dim=0)}
    torch.save(combined, output_dir / "result.pt")

    # Reconstruct SDFs
    from rdkit import Chem
    sdf_dir = output_dir / "sdf"; sdf_dir.mkdir(exist_ok=True)
    n_ok = 0
    for i, (pos, v) in enumerate(zip(all_pos, all_v)):
        try:
            mol = reconstruct.reconstruct_from_generated(pos.numpy(), v.numpy())
            if mol:
                Chem.MolToMolFile(mol, str(sdf_dir / f"{i:03d}.sdf")); n_ok += 1
        except: pass
    print(f"SDF: {n_ok}/{args.num_samples} valid")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
