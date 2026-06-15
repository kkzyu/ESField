#!/usr/bin/env python3
"""
Unified interface for molecular generation and evaluation.
Supports both EDM and SemlaFlow models.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Union, Optional

# Add the necessary paths
sys.path.append("edm")
sys.path.append("semlaflow")


def _path_for_cwd(path: Union[str, Path], cwd: Union[str, Path]) -> str:
    """Convert a repo-root-relative path into an argument for a subprocess cwd."""
    path = Path(path)
    cwd = Path(cwd)

    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == cwd.name:
        return str(Path(*path.parts[1:])) if len(path.parts) > 1 else "."
    return str(Path("..") / path)


class MolecularGenerator:
    """Unified interface for molecular generation using EDM or SemlaFlow."""

    def __init__(self, model_type: str = "edm"):
        """
        Initialize the molecular generator.

        Parameters
        ----------
        model_type : str
            Type of model to use ("edm" or "semlaflow")
        """
        self.model_type = model_type.lower()
        if self.model_type not in ["edm", "semlaflow"]:
            raise ValueError("model_type must be 'edm' or 'semlaflow'")

    def generate(
        self,
        model_config: Union[Dict, str],
        n_samples: int,
        n_atoms: int,
        use_guidance: bool = False,
        protein_path: Optional[str] = None,
        output_dir: str = "generated_molecules",
    ) -> List[str]:
        """
        Generate molecular structures using the specified model configuration.

        Parameters
        ----------
        model_config : dict or str
            Configuration for the model. This could be a path to a config file or a dictionary containing parameters.
        n_samples : int
            Number of molecular samples to generate.
        n_atoms : int
            Number of atoms in each generated molecule.
        use_guidance : bool, optional
            Whether to use guidance during generation. Default is False.
        protein_path : str or None, optional
            Path to the protein structure file to condition generation on. Default is None.
        output_dir : str, optional
            Directory to save generated molecules. Default is "generated_molecules".

        Returns
        -------
        samples : list
            A list of paths to generated molecule files.
        """

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        if self.model_type == "edm":
            return self._generate_edm(
                model_config,
                n_samples,
                n_atoms,
                use_guidance,
                protein_path,
                output_path,
            )
        elif self.model_type == "semlaflow":
            return self._generate_semlaflow(
                model_config,
                n_samples,
                n_atoms,
                use_guidance,
                protein_path,
                output_path,
            )

    def _generate_edm(
        self,
        model_config: Union[Dict, str],
        n_samples: int,
        n_atoms: int,
        use_guidance: bool,
        protein_path: Optional[str],
        output_path: Path,
    ) -> List[str]:
        """Generate molecules using EDM."""

        # Parse model config
        if isinstance(model_config, str):
            # Assume it's a checkpoint path
            checkpoint_path = model_config
            beta = 10.0
            guidance_time = 200
        else:
            # Assume it's a dictionary with parameters
            checkpoint_path = model_config.get(
                "checkpoint_path", "edm/checkpoints/model_updates_738999.ckpt"
            )
            beta = model_config.get("beta", 10.0)
            guidance_time = model_config.get("guidance_time", 200)

        cmd = [
            "python",
            "edm/conditional_sample_mols.py" if protein_path else "edm/sample_mols.py",
            "--checkpoint",
            checkpoint_path,
            "--n-molecules",
            str(n_samples),
            "--n-atoms",
            str(n_atoms),
            "--output-dir",
            str(output_path),
            "--device",
            "cuda",
        ]

        # Add protein path if provided (conditional generation)
        if protein_path:
            cmd.extend(["--protein", protein_path])

        # Add guidance parameters if requested
        if use_guidance:
            if protein_path:
                # For conditional sampling, use guidance-time and beta
                cmd.extend(
                    [
                        "--beta",
                        str(beta),
                        "--guidance-time",
                        str(guidance_time),
                        "--x0-guidance",
                    ]
                )
            else:
                # For unconditional sampling, use beta and start_t
                cmd.extend(
                    [
                        "--beta",
                        str(beta),
                        "--start_t",
                        str(guidance_time),
                        "--x0-guidance",
                    ]
                )

        # Execute command
        print(f"Running EDM generation: {' '.join(cmd)}")
        print("=" * 50)
        print("Starting EDM generation... This may take several minutes.")
        print("You should see progress updates below:")
        print("-" * 50)
        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"EDM generation failed with return code {result.returncode}"
            )

        print("-" * 50)
        print("EDM generation completed successfully!")

        generated_files = []
        for npy_file in output_path.glob("*.npy"):
            generated_files.append(str(npy_file))

        return generated_files

    def _generate_semlaflow(
        self,
        model_config: Union[Dict, str],
        n_samples: int,
        n_atoms: int,  # Kept for interface compatibility but not used
        use_guidance: bool,
        protein_path: Optional[str],
        output_path: Path,
    ) -> List[str]:
        """Generate molecules using SemlaFlow."""

        if isinstance(model_config, str):
            checkpoint_dir = model_config
        else:
            checkpoint_dir = model_config.get("checkpoint_dir", "semlaflow/checkpoints")

        semlaflow_cwd = Path("semlaflow")
        checkpoint_arg = _path_for_cwd(checkpoint_dir, semlaflow_cwd)
        proteins_arg = _path_for_cwd(
            protein_path or "edm/dataset/pdbbind", semlaflow_cwd
        )
        output_arg = _path_for_cwd(output_path, semlaflow_cwd)

        cmd = [
            "python",
            "conditional_generate_guidance.py",
            checkpoint_arg,
            proteins_arg,
            output_arg,
            "0",
            "1",
        ]
        if not use_guidance:
            cmd.append("--no-guidance")

        # Execute command from semlaflow directory
        print(f"Running SemlaFlow generation: {' '.join(cmd)}")
        print("=" * 50)
        print("Starting SemlaFlow generation... This may take several minutes.")
        print("You should see progress updates below:")
        print("-" * 50)
        result = subprocess.run(cmd, text=True, cwd=semlaflow_cwd)

        if result.returncode != 0:
            print(
                "SemlaFlow generation failed, but let's check if any files were generated..."
            )
            # Try to find any generated files even if the process failed
            generated_files = []
            for sdf_file in output_path.glob("**/*.sdf"):
                generated_files.append(str(sdf_file))

            if generated_files:
                print(f"Found {len(generated_files)} generated files despite the error")
                return generated_files
            else:
                raise RuntimeError(
                    f"SemlaFlow generation failed with return code {result.returncode}"
                )

        print("-" * 50)
        print("SemlaFlow generation completed successfully!")

        generated_files = []
        for sdf_file in output_path.glob("**/*.sdf"):
            generated_files.append(str(sdf_file))

        print(f"SemlaFlow generated files: {generated_files}")
        return generated_files


class MolecularEvaluator:
    """Unified interface for molecular evaluation."""

    def __init__(self):
        """Initialize the molecular evaluator."""
        pass

    def evaluate(self, path: str) -> Dict:
        """
        Evaluate generated molecular structures against the paper criteria.

        Parameters
        ----------
        path : str
            Path to the directory or file containing the generated molecular samples to evaluate.

        Returns
        -------
        metrics : dict
            Dictionary of evaluation metrics (e.g., validity, uniqueness, novelty, etc.).
        """

        path = Path(path)

        # Determine model type and evaluation strategy
        if "edm" in str(path):
            return self._evaluate_edm(path)
        elif "semlaflow" in str(path):
            return self._evaluate_semlaflow(path)
        else:
            # Default to EDM-style evaluation
            return self._evaluate_edm(path)

    def _evaluate_edm(self, path: Path) -> Dict:
        """Evaluate EDM files by running the analysis script."""
        print("Running EDM analysis script...")
        print("You should see progress updates below:")
        print("-" * 50)

        # Create a temporary analysis script
        analysis_script = """
#!/bin/bash
cd edm
./evaluate_edm_generated_mols.sh
"""

        # Execute the analysis and capture output while showing it in real-time
        process = subprocess.Popen(
            ["bash", "-c", analysis_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=".",
        )

        # Collect output while showing it in real-time
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())  # Show in real-time
                output_lines.append(line.rstrip())

        result_code = process.poll()

        if result_code != 0:
            print(f"EDM analysis failed with return code {result_code}")
            # Return empty metrics if analysis fails
            return {
                "vina_score": None,
                "qed": None,
                "interactions": None,
                "strain_energy": None,
                "valid_ratio": None,
            }

        # EDM script outputs metrics directly, so we can extract them from the output
        return self._extract_edm_metrics(output_lines)

    def _evaluate_semlaflow(self, path: Path) -> Dict:
        """Evaluate SemlaFlow files by running the analysis script."""
        print("Running SemlaFlow analysis script...")
        print("You should see progress updates below:")
        print("-" * 50)

        analysis_script = """
#!/bin/bash
cd semlaflow
./evaluate_semlaflow_generated_mols.sh
"""

        # Execute the analysis and capture output while showing it in real-time
        process = subprocess.Popen(
            ["bash", "-c", analysis_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=".",
        )

        # Collect output while showing it in real-time
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())  # Show in real-time
                output_lines.append(line.rstrip())

        result_code = process.poll()

        if result_code != 0:
            print(f"SemlaFlow analysis failed with return code {result_code}")
            # Return empty metrics if analysis fails
            return {
                "vina_score": None,
                "qed": None,
                "interactions": None,
                "strain_energy": None,
                "valid_ratio": None,
            }

        # SemlaFlow script now outputs metrics directly, so we can extract them
        return self._extract_semlaflow_metrics(output_lines)

    def _extract_edm_metrics(self, output_lines: List[str]) -> Dict:
        """Extract metrics from EDM script output."""
        metrics = {
            "vina_score": None,
            "qed": None,
            "interactions": None,
            "strain_energy": None,
            "valid_ratio": None,
        }

        # EDM script outputs metrics in a specific format
        for line in output_lines:
            if "Vina - Mean:" in line:
                try:
                    vina_score = float(line.split("Mean:")[1].split(",")[0].strip())
                    metrics["vina_score"] = vina_score
                except Exception:
                    pass
            elif "QED:" in line:
                try:
                    qed = float(line.split("QED:")[1].strip())
                    metrics["qed"] = qed
                except Exception:
                    pass
            elif "Interactions:" in line:
                try:
                    interactions = int(line.split("Interactions:")[1].strip())
                    metrics["interactions"] = interactions
                except Exception:
                    pass
            elif "Strain Energy:" in line:
                try:
                    strain_energy = float(line.split("Strain Energy:")[1].strip())
                    metrics["strain_energy"] = strain_energy
                except Exception:
                    pass
            elif "Valid:" in line:
                try:
                    valid_ratio = float(line.split("Valid:")[1].split("%")[0].strip())
                    metrics["valid_ratio"] = valid_ratio
                except Exception:
                    pass

        return metrics

    def _extract_semlaflow_metrics(self, output_lines: List[str]) -> Dict:
        """Extract metrics from SemlaFlow script output."""
        metrics = {
            "vina_score": None,
            "qed": None,
            "interactions": None,
            "strain_energy": None,
            "valid_ratio": None,
        }

        # SemlaFlow script outputs metrics in a different format than EDM
        for line in output_lines:
            if "Vina - Mean:" in line:
                try:
                    vina_score = float(line.split("Mean:")[1].split(",")[0].strip())
                    metrics["vina_score"] = vina_score
                except Exception:
                    pass
            elif "QED:" in line:
                try:
                    qed = float(line.split("QED:")[1].strip())
                    metrics["qed"] = qed
                except Exception:
                    pass
            elif "Interactions:" in line:
                try:
                    interactions = int(line.split("Interactions:")[1].strip())
                    metrics["interactions"] = interactions
                except Exception:
                    pass
            elif "Strain Energy:" in line:
                try:
                    strain_energy = float(line.split("Strain Energy:")[1].strip())
                    metrics["strain_energy"] = strain_energy
                except Exception:
                    pass
            elif "Valid:" in line:
                try:
                    valid_ratio = float(line.split("Valid:")[1].split("%")[0].strip())
                    metrics["valid_ratio"] = valid_ratio
                except Exception:
                    pass

        return metrics


# Convenience functions for the unified interface
def generate(
    model_config: Union[Dict, str],
    n_samples: int,
    n_atoms: int,
    use_guidance: bool = False,
    protein_path: Optional[str] = None,
    model_type: str = "edm",
    output_dir: str = "generated_molecules",
) -> List[str]:
    """
    Generate molecular structures using the specified model configuration.

    Parameters
    ----------
    model_config : dict or str
        Configuration for the model. This could be a path to a config file or a dictionary containing parameters.
    n_samples : int
        Number of molecular samples to generate.
    n_atoms : int
        Number of atoms in each generated molecule.
    use_guidance : bool, optional
        Whether to use energy guidance during generation. Default is False.
    protein_path : str or None, optional
        Path to the protein structure file to condition generation on. Default is None.
    model_type : str, optional
        Type of model to use ("edm" or "semlaflow"). Default is "edm".
    output_dir : str, optional
        Directory to save generated molecules. Default is "generated_molecules".

    Returns
    -------
    samples : list
        A list of paths to generated molecule files.
    """
    generator = MolecularGenerator(model_type)
    return generator.generate(
        model_config, n_samples, n_atoms, use_guidance, protein_path, output_dir
    )


def evaluate(path: str) -> Dict:
    """
    Evaluate generated molecular structures against the paper criteria.

    Parameters
    ----------
    path : str
        Path to the directory or file containing the generated molecular samples to evaluate.

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics (e.g., validity, uniqueness, novelty, etc.).
    """
    evaluator = MolecularEvaluator()
    return evaluator.evaluate(path)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified molecular generation interface"
    )
    parser.add_argument(
        "--model",
        choices=["edm", "semlaflow", "both"],
        default="both",
        help="Which model to run: edm, semlaflow, or both (default)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of molecules to generate (default: 10, not used by SemlaFlow)",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=30,
        help="Number of atoms per molecule (default: 30, not used by SemlaFlow)",
    )
    parser.add_argument(
        "--use-guidance",
        action="store_true",
        help="Use energy guidance during generation",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=10.0,
        help="Beta parameter for guidance (default: 10.0)",
    )
    parser.add_argument(
        "--guidance-time",
        type=int,
        default=200,
        help="Guidance time parameter (default: 200)",
    )
    parser.add_argument(
        "--protein-path",
        type=str,
        default=None,
        help="Path to an EDM protein pocket directory for conditional generation",
    )
    parser.add_argument(
        "--semlaflow-protein-dir",
        type=str,
        default="edm/dataset/pdbbind",
        help="Path to the SemlaFlow protein dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_molecules",
        help="Output directory for generated molecules",
    )

    args = parser.parse_args()
    edm_protein_path = args.protein_path or "edm/dataset/pdbbind/5zcu"

    # EDM Configuration
    edm_config = {
        "checkpoint_path": "edm/checkpoints/conditional_model_updates_487_epochs.ckpt",
        "beta": args.beta,  # Use command line argument
        "guidance_time": args.guidance_time,  # Use command line argument
    }

    # SemlaFlow Configuration (checkpoint directory path)
    semlaflow_config = "semlaflow/checkpoints"

    # Run EDM if requested
    if args.model in ["edm", "both"]:
        print("=== EDM Generation Example ===")
        try:
            samples = generate(
                model_config=edm_config,
                n_samples=args.n_samples,
                n_atoms=args.n_atoms,
                use_guidance=args.use_guidance,
                protein_path=edm_protein_path,
                model_type="edm",
                output_dir=args.output_dir,
            )
            print(f"Generated {len(samples)} EDM samples: {samples}")

            # Evaluate the generated molecules
            metrics = evaluate("edm")  # Evaluate in edm folder
            print(f"EDM Evaluation metrics: {metrics}")

        except Exception as e:
            print(f"EDM generation failed: {e}")

    # Run SemlaFlow if requested
    if args.model in ["semlaflow", "both"]:
        print("\n=== SemlaFlow Generation Example ===")
        print(
            "Note: SemlaFlow automatically determines atom count per molecule based on the size of the protein pockets from the dataset. n_samples & n_atoms are not used"
        )
        try:
            samples = generate(
                model_config=semlaflow_config,
                n_samples=args.n_samples,  # Not used by SemlaFlow but kept for interface compatibility
                n_atoms=args.n_atoms,  # Not used by SemlaFlow but kept for interface compatibility
                use_guidance=args.use_guidance,
                protein_path=args.semlaflow_protein_dir,
                model_type="semlaflow",
                output_dir=args.output_dir,
            )
            print(f"Generated {len(samples)} SemlaFlow samples: {samples}")

            # Evaluate the generated molecules
            metrics = evaluate("semlaflow")  # Evaluate in semlaflow folder
            print(f"SemlaFlow Evaluation metrics: {metrics}")

        except Exception as e:
            print(f"SemlaFlow generation failed: {e}")

    print("\n=== Generation Complete! ===")
