from pathlib import Path
from time import time
import argparse
import importlib
import os

import torch
import numpy as np
from tqdm.auto import tqdm
from src.ddpm import ContinuosDiffusionSampler
from src.ddpm import MultiCategoricalDiffusionSampler

from src.guidance_plugins import conditional_forcefield as forcefield

np.set_printoptions(precision=2, suppress=True)


def compute_repulsion_term(x_t, n_atoms, lambda_=0.05):
    # 1e-3 is a numerical floor for pairwise distances to avoid division by
    # zero when two atoms briefly coincide at noisy diffusion steps.
    d_mask = 1 - torch.eye(n_atoms, device=x_t.device)[None]
    r = x_t[:, :, None] - x_t[:, None]
    n = torch.linalg.norm(r, axis=-1, keepdims=True)
    n[n < 1e-3] = 1e-3
    d = lambda_ / torch.sqrt((torch.sum(r**2, axis=-1) + 1e-3))
    d = d * d_mask
    repulsion_term = (r / n * d[..., None]).sum(axis=-2)
    return repulsion_term


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

ATOM_DICT = np.zeros(max(ATOM_TYPE_DICT) + 1, dtype=np.float32)
INV_ATOM_DICT = np.zeros(len(ATOM_TYPE_DICT), dtype=np.float32)

for k in ATOM_TYPE_DICT:
    ATOM_DICT[k] = ATOM_TYPE_DICT[k]
    INV_ATOM_DICT[ATOM_TYPE_DICT[k]] = k


class EnergyGuidance:
    def __init__(self, denoiser, protein_path, pocket_com, guidance_type):
        self.denoiser = denoiser
        self.protein_path = protein_path
        self.com = pocket_com
        assert guidance_type in ["x0", "xt"]
        self.guidance_type = guidance_type
        if self.guidance_type == "x0":
            self.gradient_fn = self.get_gradient_x0_guidance
        elif self.guidance_type == "xt":
            self.gradient_fn = self.get_gradient_xt_guidance
        else:
            self.gradient_fn = None
        self.iterations = 0
        self.compute_jacob = 10

    def get_gradient_x0_guidance_old(self, xt, x_mask, y, y_mask, t, atom_only):
        (
            x0_hat_pos,
            x0_hat_cat,
            x0_hat_pos_jacob,
        ) = self.denoiser.get_output_and_jacobian(xt, y, t, x_mask, y_mask)
        # x0_hat_pos_jacob_t = torch.transpose(x0_hat_pos_jacob, 3, 2)
        x0_cat_mapped = INV_ATOM_DICT[
            x0_hat_cat.argmax(axis=-1, keepdims=True)
            .cpu()
            .detach()
            .numpy()
            .astype(np.int32)
        ]
        energy_batch, grad_batch = forcefield.score(
            x0_hat_pos + self.com,
            x0_cat_mapped,
            str(Path(self.protein_path) / "protein.pdb"),
            atom_only=atom_only,
            requires_grad=True,
        )

        grad_batch = torch.tensor(grad_batch).float().to(xt.device)
        grad_batch = grad_batch.view((len(grad_batch), -1))
        grad_batch = torch.matmul(grad_batch[:, None], x0_hat_pos_jacob)[:, 0]
        grad_batch = grad_batch.view(x0_hat_pos.shape)
        # grad_batch = torch.einsum("abc,abcd->abd", grad_batch, x0_hat_pos_jacob_t)
        # grad_batch_shape = grad_batch.shape
        # grad_batch = grad_batch.view(-1, 3, 1)
        # x0_hat_pos_jacob_t = x0_hat_pos_jacob_t.view(-1, 3, 3)
        # grad_batch = torch.bmm(x0_hat_pos_jacob_t, grad_batch).view(grad_batch_shape)
        return energy_batch, grad_batch, x0_hat_pos, x0_hat_cat

    def get_gradient_x0_guidance(self, xt, x_mask, y, y_mask, t, atom_only):
        with torch.no_grad():
            x0_hat_pos, x0_hat_cat = self.denoiser(xt, y, t, x_mask, y_mask)

        x0_cat_mapped = INV_ATOM_DICT[
            x0_hat_cat.argmax(axis=-1, keepdims=True)
            .cpu()
            .detach()
            .numpy()
            .astype(np.int32)
        ]
        energy_batch, grad_batch = forcefield.score(
            x0_hat_pos + self.com,
            x0_cat_mapped,
            str(Path(self.protein_path) / "protein.pdb"),
            atom_only=atom_only,
            requires_grad=True,
        )
        grad_batch = torch.tensor(grad_batch).float().to(xt.device)
        if torch.abs(grad_batch).sum() > 1e-3:
            grad_batch = grad_batch.view((len(grad_batch), -1))
            if (self.iterations % self.compute_jacob) == 0:
                x_pos = xt[..., :3]
                x_cat = xt[..., 3]
                x_shape = x_pos.shape
                size = x_shape[0]
                dim = np.prod(x_shape[1:])
                x0_hat_pos_jacob = torch.zeros(
                    (size, dim, dim), dtype=xt.dtype, device=xt.device
                )
                # Finite-difference step size for numerically approximating
                # the Jacobian of the denoiser w.r.t. positions. 1e-3 is small
                # enough to stay in the linear regime while big enough to
                # dominate float32 round-off.
                eps = 1e-3
                e_i = torch.zeros(dim, dtype=x_pos.dtype, device=x_pos.device)
                with torch.no_grad():
                    for i in range(dim):
                        e_i = e_i * 0.0
                        e_i[i] = eps
                        y_ei_p, _ = self.denoiser(
                            torch.cat(
                                (x_pos + e_i.view((1, -1, 3)), x_cat[..., None]), -1
                            ),
                            y,
                            t,
                            x_mask,
                            y_mask,
                        )
                        x0_hat_pos_jacob[:, :, i] = (y_ei_p - x0_hat_pos).view(
                            size, dim
                        ) / eps
                self.x0_hat_pos_jacob = x0_hat_pos_jacob
            else:
                x0_hat_pos_jacob = self.x0_hat_pos_jacob

            self.iterations += 1

            grad_batch = torch.matmul(grad_batch[:, None], x0_hat_pos_jacob)[:, 0]
            grad_batch = grad_batch.view(x0_hat_pos.shape)
        else:
            if self.iterations > 0:
                self.iterations += 1
        # grad_batch = torch.einsum("abc,abcd->abd", grad_batch, x0_hat_pos_jacob_t)
        # grad_batch_shape = grad_batch.shape
        # grad_batch = grad_batch.view(-1, 3, 1)
        # x0_hat_pos_jacob_t = x0_hat_pos_jacob_t.view(-1, 3, 3)
        # grad_batch = torch.bmm(x0_hat_pos_jacob_t, grad_batch).view(grad_batch_shape)
        return energy_batch, grad_batch, x0_hat_pos, x0_hat_cat

    def get_gradient_xt_guidance(self, xt, x_mask, y, y_mask, t, atom_only):
        with torch.no_grad():
            x0_hat_pos, x0_hat_cat = self.denoiser(xt, y, t, x_mask, y_mask)

        xt_pos = xt[..., :3]
        xt_cat = xt[..., 3:]
        xt_cat_mapped = INV_ATOM_DICT[xt_cat.cpu().detach().numpy().astype(np.int32)]
        energy_batch, grad_batch = forcefield.score(
            xt_pos + self.com,
            xt_cat_mapped,
            str(Path(self.protein_path) / "protein.pdb"),
            atom_only=atom_only,
            requires_grad=True,
        )
        grad_batch = torch.tensor(grad_batch).float().to(xt.device)
        return energy_batch, grad_batch, x0_hat_pos, x0_hat_cat


class Trajectory:
    def __init__(self):
        self.trajectory = []

    def add_step(self, xt):
        step = xt.detach().cpu().numpy().copy()
        step[..., -1] = INV_ATOM_DICT[step[..., -1].astype(np.int32)]
        self.trajectory.append(step)

    def save(self, fname):
        np.save(fname, self.trajectory)


def conditioned_sample(
    denoiser,
    protein,
    ddpm_con,
    ddpm_cat,
    T,
    n_examples,
    n_atoms,
    device,
    start_t=100,
    beta=0,
    guidance_on_x0=False,
    trj=None,
):
    print(trj)
    tin = time()

    num_atoms = denoiser.n_classes
    use_guidance = (beta == "auto") or (float(beta) > 0)

    # Load protein pocket
    protein_ligand = np.load(Path(protein) / "pocket.npy", allow_pickle=True)[()]
    protein_pocket = protein_ligand["ppX"][..., :4]
    protein_pocket[..., -1] = ATOM_DICT[protein_pocket[..., -1].astype(np.int64)]
    # End loading

    xt_pos = (
        torch.normal(
            torch.zeros(n_examples, n_atoms, 3),
            torch.ones(n_examples, n_atoms, 3),
        )
        .float()
        .to(device)
    )
    xt_pos = xt_pos + compute_repulsion_term(xt_pos, n_atoms)

    y_pos = torch.tensor(protein_pocket[..., :3][None]).float()
    y_cat = torch.tensor(protein_pocket[..., 3:][None]).float()

    y_pos = y_pos.repeat(n_examples, 1, 1).to(device)
    y_cat = y_cat.repeat(n_examples, 1, 1).to(device)

    y_mask = torch.zeros(y_pos.shape[:-1], dtype=torch.bool, device=device)

    com = y_pos.sum(axis=-2, keepdims=True) / (~y_mask).sum(
        -1, keepdims=True
    ).unsqueeze(-1)
    y_pos = y_pos - com

    y = torch.cat((y_pos, y_cat), dim=-1)

    xt_cat = (
        torch.randint(1, num_atoms, size=(n_examples, n_atoms, 1)).float().to(device)
    )
    xt = torch.cat((xt_pos, xt_cat), dim=-1)

    guidance_trj = Trajectory()
    guidance_trj.add_step(xt)

    if trj is not None:
        trj = np.load(trj)
        xt = trj[T - start_t]
        xt[..., 3:] = ATOM_DICT[xt[..., 3:].astype(np.int32)]
        xt = torch.tensor(xt).float().to(device)
        xt_pos = xt[..., :3]
        xt_cat = xt[..., 3:]

        guidance_trj.trajectory = [trj for trj in trj[: T - start_t + 1]]
        T = start_t

    prev_x_t_minus_1, prev_grad = None, None

    lig_mask = torch.zeros((n_examples, n_atoms), dtype=torch.bool, device=device)
    eg = EnergyGuidance(denoiser, protein, com, "x0" if guidance_on_x0 else "xt")

    adaptive_betas = torch.zeros(n_examples, dtype=torch.float32, device=device)

    for i in tqdm(range(T - 1, -1, -1)):
        grad_batch = torch.zeros_like(xt_pos)
        t = torch.from_numpy(np.array([i])).to(device)
        if (t < start_t) and use_guidance:
            # atom_only = (t > start_t // 2).item()
            atom_only = False
            energy_batch, grad_batch, x0_hat_pos, x0_hat_cat = eg.gradient_fn(
                xt, lig_mask, y, y_mask, t, atom_only=atom_only
            )

            # eps_pos = ddpm_con.predict_eps_from_x0_hat(x0_hat_pos, xt_pos, t)
            # if beta != "auto":
            # the plus sign comes from Eq.7 of
            # Energy Guided Diffusion for Generating Neurally Exciting Images
            # new_eps_pos = eps_pos - F.softmax(float(beta) * grad_batch, dim = -1)
            # grad_batch = grad_batch / torch.clip(torch.linalg.norm(grad_batch, dim=(-2,-1), keepdims=True), 1e-6)

            # Weitong's softmax
            # energy_batch = np.nan_to_num(energy_batch, copy=True, nan=np.inf, posinf=np.inf, neginf=np.inf)
            # energy_batch = torch.tensor(energy_batch).float().to(grad_batch.device)
            # smax = torch.softmax(-float(beta)*energy_batch/xt.shape[1], 0)
            # grad_softmax = torch.diag(smax) - smax[:,None] @ smax[None]
            # grad_batch = torch.einsum('mn,nkz->mkz', grad_softmax, grad_batch)
            # End Weitong's softmax

            # new_eps_pos = eps_pos + float(beta) * grad_batch
            # else:
            #     new_eps_pos = energy_optim.step(
            #         {"guidance_strength": eps_pos}, {"guidance_strength": grad_batch}
            #    )["guidance_strength"]

            # x0_hat_pos = ddpm_con.predict_x0_hat_from_eps(new_eps_pos, xt_pos, t)
        else:
            with torch.no_grad():
                x0_hat_pos, x0_hat_cat = denoiser(xt, y, t, lig_mask, y_mask)

        xt_minus_1_pos = ddpm_con.sample_diffusion_step(
            x0_hat_pos, xt_pos, t, lig_mask, deterministic=True
        )
        xt_minus_1_cat = ddpm_cat.sample_diffusion_step(
            [x0_hat_cat], xt_cat.long(), t, lig_mask
        )
        if (prev_grad is None) and (prev_x_t_minus_1 is None):
            grad_mask = torch.sum(torch.abs(grad_batch), dim=(-1, -2)) > 1e-3
            if torch.any(grad_mask):
                prev_grad = torch.clone(grad_batch)
                prev_x_t_minus_1 = torch.clone(xt_minus_1_pos.float())
        else:
            prev_grad_mask = torch.sum(torch.abs(prev_grad), dim=(-1, -2)) > 1e-3
            grad_mask = torch.sum(torch.abs(grad_batch), dim=(-1, -2)) > 1e-3
            diff_x = xt_minus_1_pos.float() - prev_x_t_minus_1
            diff_g = grad_batch - prev_grad
            num = torch.sum(diff_x * diff_x, dim=(-1, -2))
            den = torch.sum(diff_x * diff_g, dim=(-1, -2))
            den = torch.clip(den, min=1e-3)
            bb1 = num / den
            adaptive_betas = torch.clip(bb1, max=0.1)
            adaptive_betas[~torch.logical_and(grad_mask, prev_grad_mask)] = 0.0
            prev_grad = torch.clone(grad_batch) * grad_mask[:, None, None]
            prev_x_t_minus_1 = torch.clone(xt_minus_1_pos.float())

        # xt_pos = xt_minus_1_pos.float() - float(beta) / beta_den * grad_batch
        # print(adaptive_betas)
        xt_pos = xt_minus_1_pos.float() - adaptive_betas[
            :, None, None
        ] * grad_batch / torch.clip(
            torch.linalg.norm(grad_batch, dim=(-2, -1), keepdims=True), 1e-6
        )

        # xt_pos = xt_minus_1_pos.float() - 0.1 * grad_batch

        xt_cat = xt_minus_1_cat[0].argmax(axis=-1, keepdims=True)
        xt = torch.cat((xt_pos, xt_cat.float()), dim=-1)

        guidance_trj.add_step(xt)

    # this typically produces better results!
    # with torch.no_grad():
    #     x0_hat_pos, x0_hat_cat = denoiser(xt, y, t - 1, lig_mask, y_mask)

    # xt_pos = x0_hat_pos.float()
    # xt_minus_1_cat = ddpm_cat.sample_diffusion_step(
    #         [x0_hat_cat], xt_cat.long(), t-1, lig_mask
    # )
    # xt_cat = xt_minus_1_cat[0].argmax(axis=-1, keepdims=True)
    # # xt_cat = x0_hat_cat.argmax(axis=-1, keepdims=True)
    # xt = torch.cat((xt_pos, xt_cat.float()), dim=-1)
    # guidance_trj.add_step(xt)

    xt[..., :3] = xt[..., :3] + com
    total_time = time() - tin
    return xt.cpu().numpy(), guidance_trj, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--protein", type=str, default=None)
    parser.add_argument("--n-molecules", type=int, default=100)
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--beta", type=str, default="auto")
    parser.add_argument("--guidance-time", type=int, default=0)
    parser.add_argument("--x0-guidance", action="store_true", default=False)
    # To load previous trajectories
    parser.add_argument("--xt", type=str, default=None)

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

    samples, guidance_trj, ts = conditioned_sample(
        model,
        args.protein,
        ddpm_con,
        ddpm_cat,
        args.steps,
        n_mols,
        n_atoms,
        device,
        args.guidance_time,
        args.beta,
        guidance_on_x0=args.x0_guidance,
        trj=args.xt,
    )

    samples[..., 3] = INV_ATOM_DICT[samples[..., 3].astype(np.int32)]

    beta_repr = "auto" if args.beta == "auto" else f"{float(args.beta):.2f}"
    guidance_time_repr = (
        f"{args.guidance_time:03d}" if args.guidance_time > 0 else "000"
    )
    out_fn_prefix = (
        f"conditional_{n_mols:02d}_{n_atoms:02d}_{beta_repr}_{guidance_time_repr}"
    )
    fn_count = 0
    while os.path.exists(str(out_dir / f"{out_fn_prefix}_{fn_count:03d}.npy")):
        fn_count += 1
    np.save(str(out_dir / f"{out_fn_prefix}_{fn_count:03d}.npy"), samples)
    guidance_trj.save(str(out_dir / f"{out_fn_prefix}_{fn_count:03d}_trj.npy"))

    # np.save(
    #     str(out_dir / f"{out_fn_prefix}_{fn_count:03d}_trajectory.npy"),
    #     energy_trajectory,
    # )
    print(f'{str(out_dir / f"{out_fn_prefix}_{fn_count:03d}.npy")} is saved')
    print(
        f'{str(out_dir / f"{out_fn_prefix}_{fn_count:03d}_no_guidance.npy")} is saved'
    )

    # print(f'{str(out_dir / f"{out_fn_prefix}_{fn_count:03d}_trajectory.npy")} is saved')
    with open(str(out_dir / f"{out_fn_prefix}_{fn_count:03d}_time.txt"), "w") as fobj:
        fobj.write(f"{ts:f}")

    energy_batch, _ = forcefield.score(
        samples[..., :3],
        samples[..., 3, np.newaxis],
        str(Path(args.protein) / "protein.pdb"),
        requires_grad=False,
    )
    print(f"Finished generating {n_mols} samples")
