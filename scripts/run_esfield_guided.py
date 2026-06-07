#!/usr/bin/env python3
"""ESField guided generation — injects site-aware energy into PAFlow sampling.

Usage:
  python scripts/run_esfield_guided.py \
    --protein-pdb /path/to/protein.pdb \
    --site-map experiments/esfield_paflow/10gs/10gs_site_map.json \
    --potential-ckpt experiments/real_data_smoke/potential_3case/compatibility_potential_epoch_0030.pt \
    --output-dir experiments/esfield_paflow/10gs/guided \
    --volume 300 --area 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESFIELD_SRC = ROOT / "src"
PAFLOW_ROOT = Path("/root/PAFlow-main")

# --- Phase 1: Import ESField modules (ESField path first) ---
sys.path.insert(0, str(ESFIELD_SRC))

from models.potential_network import CompatibilityPotential, PotentialConfig  # noqa: E402
from models.site_features import site_type_to_index  # noqa: E402

# --- Phase 2: Import PAFlow modules (PAFlow path first to avoid conflicts) ---
sys.path.insert(0, str(PAFLOW_ROOT))
sys.path.insert(0, str(PAFLOW_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description="ESField guided PAFlow generation")
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument("--site-map", required=True, help="ESField site map JSON")
    parser.add_argument("--potential-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--volume", type=float, default=300)
    parser.add_argument("--area", type=float, default=300)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pos-grad-w", type=float, default=350.0)
    parser.add_argument("--esfield-lambda", type=float, default=0.1, help="ESField guidance weight")
    parser.add_argument("--num-steps", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(str(PAFLOW_ROOT))

    import torch
    import easydict
    torch.serialization.add_safe_globals([easydict.EasyDict])

    # Clear ESField-cached modules that conflict with PAFlow, then import PAFlow modules
    for mod_key in list(sys.modules.keys()):
        if mod_key.startswith("utils") or mod_key.startswith("models") or mod_key.startswith("datasets"):
            del sys.modules[mod_key]

    import utils.misc as misc
    import utils.transforms as trans
    from datasets.pl_data import ProteinLigandData, torchify_dict, FOLLOW_BATCH
    from utils.data import PDBProtein
    from models.molopt_score_model_guide import ScorePosNet3D_guided_flow
    from utils.evaluation import atom_num
    from torch_geometric.data import Batch
    from torch_geometric.transforms import Compose
    from torch_scatter import scatter_mean

    # Load potential with correct config (ESField modules already imported at module level)
    pot_ckpt = torch.load(args.potential_ckpt, map_location=args.device, weights_only=False)
    pot_cfg = pot_ckpt.get("config", {})
    pot_config = PotentialConfig(
        atom_embed_dim=pot_cfg.get("atom_embed_dim", 32),
        site_embed_dim=pot_cfg.get("site_embed_dim", 32),
        hidden_dim=pot_cfg.get("hidden_dim", 64),
        num_layers=pot_cfg.get("num_layers", 3),
        rbf_bins=pot_cfg.get("rbf_bins", 16),
        cutoff=pot_cfg.get("cutoff", 6.0),
        energy_clip=pot_cfg.get("energy_clip", 5.0),
    )
    potential = CompatibilityPotential(pot_config)
    potential.load_state_dict(pot_ckpt["model_state_dict"])
    potential = potential.to(args.device)
    potential.eval()
    print(f"Potential loaded (epoch {pot_ckpt.get('epoch', '?')}), config: hidden_dim={pot_config.hidden_dim}, num_layers={pot_config.num_layers}")

    # Load site map
    with open(args.site_map) as f:
        site_map = json.load(f)
    print(f"Site map: {site_map.get('n_sites', len(site_map.get('sites', [])))} sites")

    # Load PAFlow model
    config = misc.load_config(str(PAFLOW_ROOT / "configs/sampling_guide.yml"))
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom("add_aromatic")

    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    model = ScorePosNet3D_guided_flow(
        ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("PAFlow model loaded")

    # Prepare protein data
    import shutil
    models_dst = output_dir / "models"
    if not models_dst.exists():
        shutil.copytree(PAFLOW_ROOT / "models", models_dst)
    shutil.copy(PAFLOW_ROOT / "configs/sampling_guide.yml", output_dir / "sample.yml")

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(
            PDBProtein(args.protein_pdb).to_dict_atom()
        ),
        ligand_dict={
            "element": torch.empty([0,], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0,], dtype=torch.long),
        },
    )
    data = Compose([protein_featurizer])(data)

    # Predict atom numbers
    pocket_atom_num = len(data.protein_pos)
    pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
    pocket_info = torch.tensor(
        [pocket_atom_num, args.volume, args.area, pocket_size]
    ).float().to(args.device).unsqueeze(0)

    # Generate
    all_results = []
    print(f"\nGenerating {args.num_samples} molecules with ESField guidance...")
    print(f"  esfield_lambda: {args.esfield_lambda}")

    import time as time_mod
    start = time_mod.time()

    num_batches = int(torch.ceil(torch.tensor(args.num_samples / args.batch_size)).item())
    from tqdm import tqdm

    for batch_i in range(num_batches):
        n_data = min(args.batch_size, args.num_samples - batch_i * args.batch_size)

        batch = Batch.from_data_list(
            [data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH
        ).to(args.device)

        batch_protein = batch.protein_element_batch
        center_pos_batch = scatter_mean(batch.protein_pos, batch_protein, dim=0)
        pocket_size_val = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
        ligand_num_atoms = [int(atom_num.sample_atom_num(pocket_size_val).item()) for _ in range(n_data)]
        batch_ligand = torch.repeat_interleave(
            torch.arange(n_data), torch.tensor(ligand_num_atoms)
        ).to(args.device)

        init_pos = center_pos_batch[batch_ligand] + torch.randn_like(center_pos_batch[batch_ligand])
        init_v = torch.randint(0, model.num_classes, (len(batch_ligand),)).to(args.device)

        # Build ESField guidance wrapper
        class SiteGuidanceWrapper:
            def __init__(self, potential):
                self.potential = potential

        site_guidance = SiteGuidanceWrapper(potential)

        r = model.sample_guided_flow_VP(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=init_pos,
            init_ligand_v=init_v,
            batch_ligand=batch_ligand,
            num_steps=args.num_steps,
            pos_only=False,
            center_pos_mode="protein",
            noise=False,
            pos_grad_w=args.pos_grad_w,
            v_grad_w=0,
            site_guidance=site_guidance,
            site_map=site_map,
            esfield_lambda_max=args.esfield_lambda,
        )

        all_results.append({
            "pos": r["pos"].cpu(),
            "v": r["v"].cpu(),
            "pos_traj": [p.cpu() for p in r["pos_traj"]],
            "v_traj": [v.cpu() for v in r["v_traj"]],
        })

        # Free GPU memory between batches
        del r, batch, batch_protein, batch_ligand, init_pos, init_v, site_guidance
        torch.cuda.empty_cache()

    elapsed = time_mod.time() - start
    print(f"\nGuided generation done in {elapsed:.1f}s")

    # Save combined result
    # Flatten results
    all_pos = []
    all_v = []
    for res in all_results:
        n_data = res["v"].shape[0]
        # For each molecule in batch
        ligand_cum = [0]
        for nd in range(n_data):
            pass  # v is already per-molecule
        all_pos.append(res["pos"])
        all_v.append(res["v"])

    combined = {
        "pos": torch.cat(all_pos, dim=0),
        "v": torch.cat(all_v, dim=0),
    }
    torch.save(combined, output_dir / "result.pt")

    # Reconstruct molecules
    from utils import reconstruct
    all_mol_pos = [combined["pos"]]
    all_mol_v = [combined["v"]]
    n_recon, n_complete = 0, 0
    sdf_dir = output_dir / "sdf"
    sdf_dir.mkdir(exist_ok=True)

    # Determine ligand num atoms per sample for reconstruction
    total_ligand_atoms = combined["v"].shape[0]
    per_mol_atoms = args.num_samples
    # Use approximate equal splitting
    atoms_per_mol = total_ligand_atoms // args.num_samples

    for mol_idx in range(args.num_samples):
        start_a = mol_idx * atoms_per_mol
        end_a = start_a + atoms_per_mol if mol_idx < args.num_samples - 1 else total_ligand_atoms
        mol_pos = combined["pos"][start_a:end_a]
        mol_v = combined["v"][start_a:end_a]
        try:
            mol = reconstruct.reconstruct_from_generated(mol_pos, mol_v)
            if mol is not None:
                n_recon += 1
                from rdkit import Chem
                Chem.MolToMolFile(mol, str(sdf_dir / f"{mol_idx:03d}.sdf"))
                n_complete += 1
        except Exception:
            pass

    print(f"Reconstruction: {n_recon} recon, {n_complete} complete")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
