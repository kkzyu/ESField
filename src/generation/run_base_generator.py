"""CLI for dry-running or launching RepMolFlow baseline sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

from generation.adapt_generator_interface import RepMolFlowCommand, explain_repmolflow_scope


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run RepMolFlow baseline sampling.")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--n-mols", type=int, default=1000)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--n-timesteps", type=int, default=100)
    parser.add_argument("--property-name", default="alpha")
    parser.add_argument("--properties-handle-method", default="sum")
    parser.add_argument("--properties-for-sampling", type=float, default=None)
    parser.add_argument("--multiple-values-file", default=None)
    parser.add_argument("--number-of-atoms-file", default=None)
    parser.add_argument("--normalization-file-path", default=None)
    parser.add_argument("--n-atoms-per-mol", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    command = RepMolFlowCommand(
        repo_dir=Path(args.repo_dir),
        model_checkpoint=Path(args.model_checkpoint),
        output_file=Path(args.output_file),
        n_mols=args.n_mols,
        max_batch_size=args.max_batch_size,
        n_timesteps=args.n_timesteps,
        property_name=args.property_name,
        properties_handle_method=args.properties_handle_method,
        properties_for_sampling=args.properties_for_sampling,
        multiple_values_file=Path(args.multiple_values_file) if args.multiple_values_file else None,
        number_of_atoms_file=Path(args.number_of_atoms_file) if args.number_of_atoms_file else None,
        normalization_file_path=Path(args.normalization_file_path) if args.normalization_file_path else None,
        n_atoms_per_mol=args.n_atoms_per_mol,
        gpu=args.gpu,
        analyze=args.analyze,
    )
    print(explain_repmolflow_scope())
    command.run(dry_run=not args.execute)


if __name__ == "__main__":
    main()

