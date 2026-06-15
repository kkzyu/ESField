#!/usr/bin/env python3
"""
Download pretrained checkpoints for EDM and SemlaFlow from the Hugging Face
Hub into the directory structure expected by the generation scripts.

Usage:
    python download_checkpoints.py
"""

from pathlib import Path
import hashlib

from huggingface_hub import hf_hub_download

HF_REPO = "xiaoyunw/force-field-guidance-checkpoints"

# (path-in-hf-repo, destination-relative-to-this-file)
FILES = [
    (
        "edm/conditional_model_updates_487_epochs.ckpt",
        "edm/checkpoints/conditional_model_updates_487_epochs.ckpt",
    ),
    ("edm/model_updates_738999.ckpt", "edm/checkpoints/model_updates_738999.ckpt"),
    (
        "semlaflow/chpt_1743387578_299_1265.pt",
        "semlaflow/checkpoints/chpt_1743387578_299_1265.pt",
    ),
    (
        "semlaflow/model_1743387578_299_1265.pt",
        "semlaflow/checkpoints/model_1743387578_299_1265.pt",
    ),
]

EXPECTED_SHA256 = {
    "edm/conditional_model_updates_487_epochs.ckpt": (
        "0ec7c1f4e89df45418f458070a64593dc4ec210b2a3c6dc36edd6c0e228c277b"
    ),
    "edm/model_updates_738999.ckpt": (
        "b41b124204013a221f204a37d0b7a68820353b8bdc890021940a91559036ac6e"
    ),
    "semlaflow/chpt_1743387578_299_1265.pt": (
        "1881d76d1d358e3d6dcae7f29aa7b470a1efb9fc4e45b10f644f6e29c707f716"
    ),
    "semlaflow/model_1743387578_299_1265.pt": (
        "77f4666b3f4fc9eeed01b8ce11972e855f7ddd3173dc2927cfca4e0846bb67ec"
    ),
}


def sha256sum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    print(f"Downloading checkpoints from https://huggingface.co/{HF_REPO}")
    print(f"Destination: {root}")

    for hf_path, local_rel in FILES:
        dest = root / local_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {hf_path}  ->  {local_rel}")
        cached = hf_hub_download(repo_id=HF_REPO, filename=hf_path)
        actual_sha256 = sha256sum(cached)
        expected_sha256 = EXPECTED_SHA256[hf_path]
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {hf_path}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        # hf_hub_download returns a cache path; symlink or copy to the
        # destination so the generation scripts find it on the expected path.
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(cached)
        except OSError:
            import shutil

            shutil.copy2(cached, dest)

    print("Done. Checkpoints are ready.")


if __name__ == "__main__":
    main()
