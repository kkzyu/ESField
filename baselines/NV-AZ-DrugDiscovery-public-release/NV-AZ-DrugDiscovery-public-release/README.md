# Improving protein-ligand complex generation with force field guidance

Codebase for the paper [Improving protein-ligand complex generation with force
field guidance](https://doi.org/10.1186/s13321-026-01198-2), published in
Journal of Cheminformatics.

In this work, we propose a novel method to apply energy guidance solely at inference time to improve generation quality in structure based drug design without the need to retrain the backbone model itself. This modular approach allows researchers to swap out different energy guidance mechanisms as well as backbone models.

We demonstrate this on the MMFF94 energy guidance method and 2 backbone models: EDM and Semlaflow.

## Table of Contents

- [Citation](#citation)
- [Reproducibility](#reproducibility)
- [Environment Setup](#environment-setup)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Molecular Generation](#molecular-generation)
- [Reproducing Paper Tables](#reproducing-paper-tables)

## Citation

If you use this code, please cite:

```bibtex
@article{lai2026improving,
  title = {Improving protein-ligand complex generation with force field guidance},
  author = {Lai, Helen and Wang, Tingyu and Sirelkhatim, Hassan and Eaton, Joe and Huang, Howard and Rees, Brad and Engkvist, Ola and Janet, Jon Paul and Wang, Xiaoyun and Tibo, Alessandro},
  journal = {Journal of Cheminformatics},
  volume = {18},
  pages = {55},
  year = {2026},
  doi = {10.1186/s13321-026-01198-2},
  url = {https://doi.org/10.1186/s13321-026-01198-2}
}
```

## Reproducibility

This repository supports two reproducibility levels:

- **Fresh-clone demo**: uses the tracked `5zcu` example protein pocket, public
  checkpoints, and the commands below. This verifies installation, checkpoint
  loading, EDM generation, SemlaFlow generation, and force-field guidance.
- **Paper tables**: requires the full preprocessed PDBBind test inputs and the
  generated/evaluated result artifacts used in the paper. Glide-score
  reproduction additionally requires a valid Schrödinger license; otherwise use
  the precomputed Glide files from the paper artifact bundle.

The checkpoint downloader verifies SHA256 checksums for the public weights. If
any hosted checkpoint changes, `python download_checkpoints.py` fails before
placing the file in the expected local path.

## Environment Setup

### Prerequisites

- A CUDA-compatible GPU (recommended; CPU execution works but is slow)
- One of:
  - [Conda / Mamba](https://docs.conda.io/) (recommended), or
  - Docker + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU passthrough

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate edm
```

This is the primary, reproducible install path. `environment.yml` pins Python 3.11,
PyTorch 2.3+ with CUDA 12.1, RDKit, OpenBabel, and the full scientific stack.

### Option B — Python installer (fallback)

If the conda solve is slow or you prefer `pip`:

```bash
conda create -n edm python=3.11 -y
conda activate edm
python install_dependencies.py
```

### Option C — Docker

The repo ships a public-usable `Dockerfile` that builds from
[`mambaorg/micromamba`](https://hub.docker.com/r/mambaorg/micromamba) and installs
the full `environment.yml`:

```bash
# Build
docker build -t ffguidance .

# Run with GPU access and the repo bind-mounted
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    ffguidance
```

Inside the container the `edm` env is already active.

## Pretrained Checkpoints

Model weights for EDM (conditional + unconditional) and SemlaFlow are
published on the Hugging Face Hub:
👉 [`xiaoyunw/force-field-guidance-checkpoints`](https://huggingface.co/xiaoyunw/force-field-guidance-checkpoints)

Download everything with one command from the repo root:

```bash
python download_checkpoints.py
```

The script places files into `edm/checkpoints/` and `semlaflow/checkpoints/`,
matching the paths expected by the generation scripts below. Downloaded files
are checksum-verified.

**Files fetched** (~193 MB total):

| File | Size | Purpose |
| ---- | ---- | ------- |
| `edm/checkpoints/conditional_model_updates_487_epochs.ckpt` | 21 MB | EDM conditional (pocket-aware) |
| `edm/checkpoints/model_updates_738999.ckpt` | 21 MB | EDM unconditional |
| `semlaflow/checkpoints/model_1743387578_299_1265.pt` | 151 MB | SemlaFlow weights |
| `semlaflow/checkpoints/chpt_1743387578_299_1265.pt` | 1 KB | SemlaFlow config metadata |

The SemlaFlow optimizer state (~452 MB) needed only for resuming training
is not distributed publicly — contact the authors if you need it.

## Molecular Generation

The main interface for molecular generation is `molecular_generation.py`, which supports both EDM and SemlaFlow models.

### Basic Usage

```bash
# Generate molecules using EDM model
python molecular_generation.py --model edm --n-samples 10 --n-atoms 30 --use-guidance

# Generate molecules using SemlaFlow model
python molecular_generation.py --model semlaflow --use-guidance

# Generate molecules using both models
python molecular_generation.py --model both --n-samples 10 --n-atoms 30 --use-guidance
```

**Note**: SemlaFlow model doesn't use `--n-samples` or `--n-atoms` parameters. See Semlaflow section.

### Advanced Options

```bash
python molecular_generation.py \
  --model edm \
  --n-samples 100 \
  --n-atoms 40 \
  --use-guidance \
  --beta 10.0 \
  --guidance-time 200 \
  --protein-path edm/dataset/pdbbind/5zcu \
  --output-dir generated_molecules
```

### Parameters

- `--model`: Choose model type (`edm`, `semlaflow`, or `both`)
- `--n-samples`: Number of molecules to generate (default: 10)
- `--n-atoms`: Number of atoms per molecule (default: 30)
- `--use-guidance`: Enable energy guidance during generation
- `--beta`: Beta parameter for guidance (default: 10.0)
- `--guidance-time`: Guidance time parameter (default: 200)
- `--protein-path`: Path to protein for conditional generation
- `--output-dir`: Output directory for generated molecules

### Direct Model Usage

#### EDM Model

```bash
# Basic generation
cd edm
python sample_mols.py \
  --checkpoint checkpoints/model_updates_738999.ckpt \
  --n-molecules 10 \
  --n-atoms 40 \
  --output-dir raw_mols \
  --device cuda

# With energy guidance
python sample_mols.py \
  --checkpoint checkpoints/model_updates_738999.ckpt \
  --n-molecules 10 \
  --n-atoms 40 \
  --output-dir raw_mols \
  --beta 10 \
  --start_t 200 \
  --device cuda \
  --x0-guidance

# Conditional generation (protein pockets)
python conditional_sample_mols.py \
  --checkpoint checkpoints/conditional_model_updates_487_epochs.ckpt \
  --protein dataset/pdbbind/5zcu \
  --n-molecules 10 \
  --n-atoms 30 \
  --output-dir raw_mols \
  --beta=10 \
  --guidance-time=200 \
  --device cuda \
  --x0-guidance
```

**Note**: EDM's `conditional_sample_mols.py` requires user input for number of molecules to generate and atom count per molecule.

**EDM Command Line Parameters:**

**Unconditional Generation (`sample_mols.py`):**

- `--checkpoint`: Path to model checkpoint file
- `--device`: Device to use ("cpu" or "cuda", default: "cpu")
- `--n-molecules`: Number of molecules to generate (default: 100)
- `--n-atoms`: Number of atoms per molecule (default: None, must be specified)
- `--steps`: Number of diffusion steps (default: 1000)
- `--output-dir`: Output directory for generated molecules
- `--beta`: Guidance strength parameter (default: 0, no guidance)
- `--start_t`: Step to start guidance (default: 0)
- `--x0-guidance`: Enable x0 guidance (default: False)

**Conditional Generation (`conditional_sample_mols.py`):**

- `--checkpoint`: Path to model checkpoint file
- `--device`: Device to use ("cpu" or "cuda", default: "cpu")
- `--protein`: Path to protein directory for conditional generation
- `--n-molecules`: Number of molecules to generate (default: 100)
- `--n-atoms`: Number of atoms per molecule (default: None, must be specified)
- `--steps`: Number of diffusion steps (default: 1000)
- `--output-dir`: Output directory for generated molecules
- `--beta`: Guidance strength parameter (default: "auto")
- `--guidance-time`: Step to start guidance (default: 0)
- `--x0-guidance`: Enable x0 guidance (default: False)
- `--xt`: Path to previous trajectory file (optional)

#### SemlaFlow Model

```bash
cd semlaflow
python conditional_generate_guidance.py \
  checkpoints/ \
  ../edm/dataset/pdbbind \
  generated_molecules \
  0 \
  1 \
  --protein-ids 5zcu
```

**Note**: SemlaFlow's `conditional_generate_guidance.py` uses positional arguments and has **fixed generation parameters**:

- Attempts to generate **128 molecules per protein** (not configurable)
- Atom count per molecule is automatically determined from the native ligand size in the dataset (capped at 128 atoms max)
- Processes all proteins that are present in both `semlaflow/cache/pdbbind_test.npy` and the provided protein directory
- Output count varies: Typically 80-90% of generated molecules pass validation (around 100-115 molecules per protein)

The command line parameters are:

- `checkpoint`: Path to model checkpoints
- `proteins_path`: Directory containing per-protein folders with `protein.pdb` files
- `output_path`: Directory to save generated molecules
- `gpu_id`: GPU ID to use (0 for single GPU)
- `num_gpus`: Number of GPUs (1 for single GPU)
- `--protein-ids`: Optional list of protein IDs to run, for example `--protein-ids 5zcu`
- `--max-proteins`: Optional cap on the number of proteins after filtering

### Converting Point Clouds to SDF Files

```bash
cd edm
python src/guidance_plugins/utils/cloud2mol.py \
  raw_mols/10_40_000.npy \
  molecules_10_atoms_40.sdf
```

## Reproducing Paper Tables

To reproduce tables from newly generated molecules:

1. **Generate molecules** using the methods above
2. **Run the table generation script:**

To reproduce the exact paper tables:

1. Download the paper result artifact bundle.
2. Put generated/evaluated results in `paper_results/`.
3. Put native-label/reference files in `pdb_dir/`.
4. Run the table generation script:

```bash
python generate_paper_tables.py paper_results pdb_dir
```

This script will:

- Read generated molecular structures
- Calculate various molecular properties and metrics
- Generate tables with the same format as the paper
- Provide evaluation metrics including Vina scores, QED, and other molecular properties

### Evaluation Metrics

The evaluation includes:

- **QED**: Quantitative Estimate of Drug-likeness
- **PBR**: PoseBusters pass ratio — geometric / chemical plausibility of poses
- **BNC**: Better-than-Native Count — fraction of generated ligands with a better docking score than the native ligand
- **Strain Energy**: MMFF94 force-field energies of the generated ligands
- **Vina Score**: AutoDock Vina docking score (lower is better)
- **Glide Score**: Schrödinger Glide docking score (lower is better; requires a Schrödinger license — scores for the paper were pre-computed)
