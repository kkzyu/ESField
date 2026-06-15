# SemlaFlow — Force-Field-Guided Conditional Generation

This directory contains the SemlaFlow backbone and scripts used for conditional
ligand generation with MMFF94 force-field guidance, as described in the paper.

See the [top-level README](../README.md) for the project overview, paper reference,
and environment setup.

## Layout

- `conditional_generate_guidance.py` — main entry point for generating ligands
  conditional on protein pockets, with optional force-field guidance.
- `postoptimization.py` — post-hoc MMFF94 minimization of already-generated ligands.
- `evaluate_semlaflow_generated_mols.sh` — end-to-end evaluation pipeline
  (generation → fragment removal → scoring).
- `checkpoints/` — SemlaFlow model weights (see top-level README for download).
- `cache/` — preprocessed protein / pocket inputs.
- `src/` — model, data, and inference code.

## Conditional generation

```bash
python conditional_generate_guidance.py \
    <checkpoint_dir> <proteins_path> <output_dir> <gpu_id> <num_gpus> [--no-guidance]
```

**Positional arguments:**

| Arg             | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `checkpoint`    | Path to the directory containing model checkpoints           |
| `proteins_path` | Path to the preprocessed protein inputs (typically `cache/`) |
| `output_path`   | Directory where generated ligands will be written            |
| `gpu_id`        | GPU index to use for this worker                             |
| `num_gpus`      | Total number of GPU workers in the run                       |

**Optional flag:**

- `--no-guidance` — disable force-field guidance (ablation / baseline).

**Example — single GPU:**

```bash
python conditional_generate_guidance.py checkpoints/ cache/ generated_molecules 0 1
```

**Example — multi-GPU (N GPUs):** launch `N` processes, one per GPU. For GPU
index `i` in `{0, ..., N-1}`:

```bash
python conditional_generate_guidance.py checkpoints/ cache/ generated_molecules i N
```

### Fixed generation parameters

`conditional_generate_guidance.py` processes every protein in `proteins_path`
automatically. Per protein it attempts **128 generations**; atom counts are
derived from the native ligand size (capped at 128 atoms). Roughly 80–90 % of
generated molecules typically pass validation.

## Post-hoc MMFF94 optimization

For already-generated ligands in SDF form, `postoptimization.py` runs MMFF94
minimization:

```bash
python postoptimization.py <protein.pdb> <ligands.sdf> <output.sdf>
```

## End-to-end evaluation

`evaluate_semlaflow_generated_mols.sh` chains generation, fragment removal, and
scoring. Edit paths at the top of the script before running.
