import importlib
import argparse
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.ddpm import ContinuosDiffusionSampler
from src.ddpm import MultiCategoricalDiffusionSampler
from src.guidance_plugins import forcefield

ATOM_TYPE_DICT = {
    1: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    14: 6,
    15: 7,
    16: 8,
    17: 9,
    35: 10,
    53: 11,
    0: 0,
}

INV_ATOM_DICT = np.zeros(len(ATOM_TYPE_DICT), dtype=np.float32)
for k in ATOM_TYPE_DICT:
    INV_ATOM_DICT[ATOM_TYPE_DICT[k]] = k


def compute_repulsion_term(x_t, n_atoms, lambda_=0.05):
    # Clamp/epsilon values (1e-3) guard against division by zero when two
    # sampled atom coordinates coincide during the early (noisy) diffusion
    # steps; keeps the repulsion term finite without affecting well-separated
    # atoms.
    d_mask = 1 - torch.eye(n_atoms, device=x_t.device)[None]
    r = x_t[:, :, None] - x_t[:, None]
    n = torch.linalg.norm(r, axis=-1, keepdims=True)
    n[n < 1e-3] = 1e-3
    d = lambda_ / torch.sqrt((torch.sum(r**2, axis=-1) + 1e-3))
    d = d * d_mask
    repulsion_term = (r / n * d[..., None]).sum(axis=-2)
    return repulsion_term


def sample(
    denoiser,
    ddpm_con,
    ddpm_cat,
    T,
    n_examples,
    n_atoms,
    device,
    start_t=100,
    beta=0,
    guidance_on_x0=False,
):
    num_atoms = denoiser.n_classes
    use_guidance = beta > 0

    xt_pos = (
        torch.normal(
            torch.zeros(n_examples, n_atoms, 3),
            torch.ones(n_examples, n_atoms, 3),
        )
        .float()
        .to(device)
    )
    xt_pos = xt_pos + compute_repulsion_term(xt_pos, n_atoms)
    xt_pos = xt_pos - xt_pos.mean(axis=-2, keepdims=True)
    xt_cat = (
        torch.randint(1, num_atoms, size=(n_examples, n_atoms, 1)).float().to(device)
    )
    xt = torch.cat((xt_pos, xt_cat), dim=-1)
    mask = torch.zeros((n_examples, n_atoms), dtype=torch.bool).to(device)
    mean_energy = np.inf
    energy_trajectory = []
    for i in tqdm(range(T - 1, 0, -1)):
        t = torch.from_numpy(np.array([i])).to(device)

        if (t < start_t) and use_guidance:
            if guidance_on_x0:
                (
                    x0_hat_pos,
                    x0_hat_cat,
                    x0_hat_pos_jacob,
                ) = denoiser.get_output_and_jacobian(xt, t, mask)
                x0_hat_pos_jacob_t = torch.transpose(x0_hat_pos_jacob, 3, 2)
                x0_hat_pos = x0_hat_pos - x0_hat_pos.mean(axis=-2, keepdims=True)

                x0_cat_mapped = INV_ATOM_DICT[
                    x0_hat_cat.argmax(axis=-1, keepdims=True)
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.int32)
                ]
                energy_batch, grad_batch = forcefield.score(
                    x0_hat_pos, x0_cat_mapped, requires_grad=True
                )
            else:
                with torch.no_grad():
                    x0_hat_pos, x0_hat_cat = denoiser(xt, t, mask)
                x0_hat_pos = x0_hat_pos - x0_hat_pos.mean(axis=-2, keepdims=True)
                xt_cat_mapped = INV_ATOM_DICT[
                    xt_cat.cpu().detach().numpy().astype(np.int32)
                ]
                energy_batch, grad_batch = forcefield.score(
                    xt_pos, xt_cat_mapped, requires_grad=True
                )

            mean_energy = np.nanmean(energy_batch)
            energy_trajectory.append(mean_energy)

            grad_batch = torch.tensor(grad_batch).float().to(device)

            if guidance_on_x0:
                grad_batch_shape = grad_batch.shape
                grad_batch = grad_batch.view(-1, 3, 1)
                x0_hat_pos_jacob_t = x0_hat_pos_jacob_t.view(-1, 3, 3)
                grad_batch = torch.bmm(x0_hat_pos_jacob_t, grad_batch).view(
                    grad_batch_shape
                )

            recon_rate = len(energy_batch) / len(xt_pos)
            print(
                f"mean energy at time {t.item()} with beta: {(beta)}: {mean_energy}, reconstruction rate : {recon_rate}"
            )
            if sum(sum(xt_cat)) == 0:
                raise ValueError("All atom types are zero")

            eps_pos = ddpm_con.predict_eps_from_x0_hat(x0_hat_pos, xt_pos, t)
            new_eps_pos = eps_pos - F.softmax(beta * grad_batch, dim=-1)
            x0_hat_pos = ddpm_con.predict_x0_hat_from_eps(new_eps_pos, xt_pos, t)
        else:
            with torch.no_grad():
                x0_hat_pos, x0_hat_cat = denoiser(xt, t, mask)
            x0_hat_pos = x0_hat_pos - x0_hat_pos.mean(axis=-2, keepdims=True)

        xt_minus_1_pos = ddpm_con.sample_diffusion_step(
            x0_hat_pos, xt_pos, t, mask, deterministic=True
        )
        xt_minus_1_cat = ddpm_cat.sample_diffusion_step(
            [x0_hat_cat], xt_cat.long(), t, mask
        )

        xt_pos = xt_minus_1_pos.float()
        xt_pos = xt_pos - xt_pos.mean(axis=-2, keepdims=True)
        xt_cat = xt_minus_1_cat[0].argmax(axis=-1, keepdims=True)
        xt = torch.cat((xt_pos, xt_cat.float()), dim=-1)

    # this typically produces better results!
    with torch.no_grad():
        x0_hat_pos, x0_hat_cat = denoiser(xt, t, mask)

    xt_pos = x0_hat_pos - x0_hat_pos.mean(axis=-2, keepdims=True)
    xt_cat = x0_hat_cat.argmax(axis=-1, keepdims=True)
    xt = torch.cat((xt_pos, xt_cat.float()), dim=-1)

    return xt.cpu().detach().numpy(), energy_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-molecules", type=int, default=100)
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--start_t", type=int, default=0)
    parser.add_argument("--x0-guidance", action="store_true", default=False)

    args = parser.parse_args()

    n_mols = args.n_molecules
    n_atoms = args.n_atoms

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving the sampled mols in {str(out_dir)}")

    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    model_name = checkpoint["model_type"]
    model_class = getattr(
        importlib.import_module(f"src.models.{model_name.lower()}.model"), model_name
    )
    device = args.device

    model = model_class.load_from_checkpoint(checkpoint, device)
    print(f"{model_name} model loaded on {device}")

    model = model.eval()

    ddpm_con = ContinuosDiffusionSampler(args.steps).to(device)
    ddpm_cat = MultiCategoricalDiffusionSampler(
        args.steps, categorical_dims=[model.n_classes]
    ).to(device)

    print(f"Generating {n_mols} mols with {n_atoms} atoms")
    if args.x0_guidance:
        print("Using guidance on x0")

    samples, energy_trajectory = sample(
        model,
        ddpm_con,
        ddpm_cat,
        args.steps,
        n_mols,
        n_atoms,
        device,
        args.start_t,
        args.beta,
        guidance_on_x0=args.x0_guidance,
    )
    samples[..., 3] = INV_ATOM_DICT[samples[..., 3].astype(np.int32)]
    out_fn_prefix = (
        f"unconditional_{n_mols:02d}_{n_atoms:02d}_{args.beta:.2f}_{args.start_t}"
    )
    fn_count = 0
    while os.path.exists(str(out_dir / f"{out_fn_prefix}_{fn_count:03d}.npy")):
        fn_count += 1
    np.save(str(out_dir / f"{out_fn_prefix}_{fn_count:03d}.npy"), samples)
    np.save(
        str(out_dir / f"{out_fn_prefix}_{fn_count:03d}_trajectory.npy"),
        energy_trajectory,
    )
    print(f'{str(out_dir / f"{out_fn_prefix}_{fn_count:03d}.npy")} is saved')
    print(f'{str(out_dir / f"{out_fn_prefix}_{fn_count:03d}_trajectory.npy")} is saved')

    energy_batch, _ = forcefield.score(
        samples[..., :3], samples[..., 3, np.newaxis], requires_grad=False
    )
    print(f"Mean energy : {np.mean(energy_batch)}")
    print(f"Finished generating {n_mols} samples")
