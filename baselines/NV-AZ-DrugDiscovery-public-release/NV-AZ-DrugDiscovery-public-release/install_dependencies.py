#!/usr/bin/env python3
"""
Dependency installer for the force-field-guided molecular generation codebase.

Installs everything needed to run EDM, SemlaFlow, guidance plugins, and the
paper-table evaluation scripts.

Usage:
    conda create -n edm python=3.11
    conda activate edm
    python install_dependencies.py
"""

import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Installing {description}...")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {description}")
        print(f"Error: {e.stderr}")
        return False


def main():
    print("Force-field-guided generation: dependency installer")
    print("=" * 52)

    # Check if conda is available
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Conda not found. Please install conda first.")
        sys.exit(1)

    # OpenBabel ships cleanly only via conda
    if not run_command("conda install -c conda-forge openbabel -y", "OpenBabel"):
        print(
            "Failed to install OpenBabel. Please install manually: conda install -c conda-forge openbabel"
        )

    # Pip packages (excluding torch — installed separately with CUDA index)
    packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.60.0",
        "rdkit>=2023.0.0",
        "func_timeout==4.3.5",
        "natsort",
        "matplotlib",
        "pandas",
        "seaborn",
        "Pillow",
        "prolif",
    ]

    for package in packages:
        if not run_command(f"pip install {package}", package):
            print(f"Failed to install {package}")

    # PyTorch with CUDA 12.1
    if not run_command(
        "pip install torch --index-url https://download.pytorch.org/whl/cu121",
        "PyTorch with CUDA",
    ):
        print("Failed to install PyTorch. Please install manually.")

    print("\n" + "=" * 52)
    print("Installation complete.")


if __name__ == "__main__":
    main()
