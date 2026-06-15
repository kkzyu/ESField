from pathlib import Path
import argparse
import random
import os

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import numpy as np
import torch


from src.data.interpolate import NoiseSampler
from src.data.vocabulary import Vocabulary
from src.inference.utils import (
    cloud2mol,
    prepare_input,
    _uniform_sample_step,
    prepare_protein_input,
)
from src.train.checkpoint import CheckpointManager
from src.utils.torch_forcefield import TorchMMFF94

RDLogger.DisableLog("rdApp.*")
np.set_printoptions(precision=2, suppress=True)


def set_seed(seed: int = 42):
    random.seed(seed)  # Python built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (CUDA)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = (
        False  # Disables optimization that may introduce randomness
    )


def compute_energy(force_field, x, ligand):
    if ligand is None:
        return False
    try:
        loss, ligand_loss = force_field.forward(x, ligand)
        loss.backward()
        return True
    except Exception:
        return False


def to_mols(x, h, c, e, mask, com, vocab):
    _mols = {
        "x": x.detach().cpu() + com.cpu(),
        "c": torch.cat(
            (
                h.detach().argmax(-1, keepdims=True),
                c.detach().argmax(-1, keepdims=True),
            ),
            -1,
        ).cpu(),
        "a": e.detach().argmax(-1).cpu(),
        "mask": mask.detach().cpu(),
    }
    _mols = vocab.decode(_mols)
    mols = []
    for _x, _c, _a in zip(_mols["x"], _mols["c"], _mols["a"]):
        _mol = cloud2mol(_x, _c, _a)
        if _mol is not None:
            try:
                _mol.UpdatePropertyCache()
                Chem.GetSSSR(_mol)
                if len(Chem.GetMolFrags(_mol)) > 1:
                    _mol = None
                if _mol is not None:
                    _mol_tmp = Chem.Mol(_mol)
                    Chem.SanitizeMol(_mol_tmp)
            except Exception:
                _mol = None
        mols.append(_mol)
    return mols


def generate(
    model,
    mols_to_sample,
    batch_prot,
    steps=100,
    batch_size=4,
    verbose=True,
    device="cuda",
    protein=None,
    use_guidance=True,
):
    # model, mols_to_sample
    vocab = Vocabulary()
    protein_vocabulary = Vocabulary({"H", "C", "N", "O", "S"})
    try:
        enc_batch_prot = protein_vocabulary.encode(batch_prot)
    except Exception:
        return []
    den = torch.clamp((~enc_batch_prot["mask"]).sum(-1, keepdims=True), min=1.0)[
        ..., None
    ]
    com = enc_batch_prot["x"].sum(1, keepdims=True) / den
    enc_batch_prot["x"] = (enc_batch_prot["x"] - com) * (
        ~enc_batch_prot["mask"][..., None]
    )
    px, ph, _, _, pmask = prepare_protein_input(enc_batch_prot, protein_vocabulary)

    pxb = px.to(device)
    phb = ph.to(device)
    pmaskb = pmask.to(device)

    scale_ot_factor = 0.25645364536453646
    # scale_ot_factor = 0.2
    noise_sampler = NoiseSampler(
        vocab,
        scale_ot=True,
        scale_ot_factor=scale_ot_factor,
        sample_charges=False,
        symmetric_adj=False,
    )

    atoms = np.where(mols_to_sample > 0)[0]
    examples = mols_to_sample[atoms]

    time_points = (1 - np.geomspace(0.01, 1.0, steps + 1))[::-1]
    step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]

    pbar = zip(atoms, examples)
    if verbose:
        pbar = tqdm(pbar, total=len(atoms))

    force_field = TorchMMFF94(protein=protein, device=device)

    generated_mols = []

    for param in model.parameters():
        param.requires_grad_(False)

    for num_atoms, num_examples in pbar:
        inference_iterations = int(np.ceil(num_examples / batch_size))
        steps = 100
        time_points = (1 - np.geomspace(0.01, 1.0, steps + 1))[::-1]
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        # step_sizes.append(1 - np.sum(step_sizes))

        for ii in tqdm(range(inference_iterations)):
            li = ii * batch_size
            ri = min((ii + 1) * batch_size, num_examples)
            bs = ri - li
            px = pxb.repeat(bs, 1, 1)
            ph = phb.repeat(bs, 1, 1)
            pmask = pmaskb.repeat(bs, 1)

            times = torch.zeros(bs, device=device)
            enc_noise = noise_sampler.sample_batch([num_atoms] * bs)
            for k in enc_noise:
                enc_noise[k] = enc_noise[k].to(device)
            x, h, _, e, mask = prepare_input(enc_noise, times, vocab)

            if model.params["self_conditioning"]:
                given_x = torch.zeros_like(x)
                given_h = torch.zeros_like(h[..., :-1])
                given_e = torch.zeros_like(e)

            loss = None
            for step_size in step_sizes:
                # with torch.no_grad():
                x.requires_grad_(True)
                if model.params["self_conditioning"]:
                    x_p, h_p, c_p, e_p = model(
                        x,
                        h,
                        e,
                        mask,
                        given_x=given_x,
                        given_h=given_h,
                        given_e=given_e,
                        px=px,
                        ph=ph,
                        pmask=pmask,
                    )
                else:
                    x_p, h_p, c_p, e_p = model(x, h, e, mask, px=px, ph=ph, pmask=pmask)

                h_probs = torch.softmax(h_p, -1)
                c_probs = torch.softmax(c_p, -1)
                e_probs = torch.softmax(e_p, -1)

                given_x = x_p.detach()
                given_h = h_probs.detach()
                given_e = e_probs.detach()

                mols = to_mols(x_p, h_probs, c_probs, e_probs, mask, com, vocab)
                e_grads = []
                for mi, mol in enumerate(mols):
                    if (mol is None) or (not use_guidance):
                        grad = torch.zeros_like(x_p[0:1])
                    else:
                        try:
                            force_field.setup(mol)
                            loss, ligand_loss = force_field.forward(
                                (x_p[mi] + com[0].to(device)) * vocab.std
                            )
                            loss.backward(retain_graph=(len(x_p) > 1))
                            grad = x.grad
                            if grad is not None:
                                grad = grad[mi : mi + 1]
                                grad = grad / (torch.linalg.norm(grad) + 1e-8)
                            else:
                                grad = torch.zeros_like(x_p[0:1])
                            x.grad = None
                            x_p.grad = None
                        except Exception:
                            grad = torch.zeros_like(x_p[0:1])
                    e_grads.append(grad)
                e_grads = torch.cat(e_grads, 0) / vocab.std
                xv = (x_p - x) / (1 - times.view(-1, 1, 1))
                x = x + step_size * xv - e_grads

                h[..., 1:] = _uniform_sample_step(h[..., 1:], h_probs, times, step_size)
                e = _uniform_sample_step(e, e_probs, times, step_size)

                times = times + float(step_size)
                h[..., 0] = times.view(-1, 1).repeat((1, h.shape[1]))
                x = x.detach()

            generated_mols += to_mols(x_p, h_probs, c_probs, e_probs, mask, com, vocab)

    return generated_mols


def get_args():
    parser = argparse.ArgumentParser(
        description="Process protein generation arguments."
    )

    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint.")
    parser.add_argument("proteins_path", type=str, help="Path to the input proteins.")
    parser.add_argument(
        "output_path", type=str, help="Path where outputs will be saved."
    )
    parser.add_argument("gpu_id", type=int, help="ID of the GPU to use.")
    parser.add_argument("num_gpus", type=int, help="Number of GPUs to use.")
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        default=False,
        help="Disable guidance if this flag is set.",
    )
    parser.add_argument(
        "--protein-ids",
        nargs="+",
        default=None,
        help="Optional protein IDs to generate. Defaults to all IDs with protein.pdb files.",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Optional cap on the number of proteins processed after filtering.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    chpt_path = Path(args.checkpoint)
    protein_prefix = Path(args.proteins_path)
    output_path = args.output_path
    gpu_id = args.gpu_id
    n_gpus = args.num_gpus
    use_guidance = not args.no_guidance

    os.makedirs(output_path, exist_ok=True)
    set_seed(42)
    device = "cuda"
    model, _, chpt_mgr = CheckpointManager.restore_last_checkpoint(chpt_path)
    model = model.to(device)
    dataset = np.load("cache/pdbbind_test.npy", allow_pickle=True)[()]
    pids = sorted(list(dataset.keys()))
    if args.protein_ids is not None:
        requested_pids = set(args.protein_ids)
        missing_from_cache = sorted(requested_pids - set(pids))
        if missing_from_cache:
            raise ValueError(
                "Requested protein IDs are missing from cache/pdbbind_test.npy: "
                + ", ".join(missing_from_cache)
            )
        pids = [pid for pid in pids if pid in requested_pids]

    missing_protein_files = [
        pid for pid in pids if not (protein_prefix / pid / "protein.pdb").exists()
    ]
    if missing_protein_files:
        print(
            "Skipping protein IDs without protein.pdb under "
            f"{protein_prefix}: {missing_protein_files[:10]}"
            + (" ..." if len(missing_protein_files) > 10 else "")
        )
        pids = [pid for pid in pids if pid not in set(missing_protein_files)]

    if args.max_proteins is not None:
        pids = pids[: args.max_proteins]
    if not pids:
        raise ValueError(
            f"No proteins to process. Check --proteins_path ({protein_prefix}) "
            "or pass --protein-ids for IDs present in both the cache and protein directory."
        )

    batch = np.array_split(np.arange(len(pids)), n_gpus)[gpu_id]
    pids = [pids[i] for i in batch]
    print(pids)

    for pid in pids:
        protein_pocket = dataset[pid]["protein"]
        if len(protein_pocket[0]) > 500:
            idx = np.argsort((protein_pocket[0] ** 2).sum(-1))
            protein_pocket[0] = protein_pocket[0][idx[:500]]
            protein_pocket[1] = protein_pocket[1][idx[:500]]

        native_ligand_size = len(dataset[pid]["ligand"][0])
        if native_ligand_size > 128:
            native_ligand_size = 128
        os.makedirs(f"{output_path}/{pid}", exist_ok=True)
        if os.path.exists(f"{output_path}/{pid}/generated_mols.sdf"):
            continue
        print(pid, len(protein_pocket[0]), native_ligand_size)

        mols_to_sample = np.zeros(native_ligand_size + 1, dtype=np.int32)
        mols_to_sample[native_ligand_size] = 128

        xprot, cprot = (
            torch.tensor(protein_pocket[0][None]).float(),
            torch.tensor(protein_pocket[1][None]).long(),
        )
        batch_prot = {
            "x": xprot,
            "c": cprot,
            "a": None,
            "mask": torch.zeros_like(xprot[..., 0]).bool(),
        }
        protein = Chem.MolFromPDBFile(
            str(protein_prefix / f"{pid}/protein.pdb"), sanitize=False, removeHs=False
        )
        protein = Chem.AddHs(protein, addCoords=True)
        Chem.GetSSSR(protein)

        mols = generate(
            model,
            mols_to_sample,
            batch_prot,
            steps=100,
            batch_size=1,
            verbose=True,
            device=device,
            protein=protein,
            use_guidance=use_guidance,
        )
        writer = Chem.SDWriter(f"{output_path}/{pid}/generated_mols.sdf")
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            mol.SetProp("_Name", str(i))
            writer.write(mol)
        writer.close()
