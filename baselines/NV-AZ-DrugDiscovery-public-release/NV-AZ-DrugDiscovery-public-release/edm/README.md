## Installation

### Quick Install (Recommended)
```bash
# Create and activate conda environment
conda create -n edm python=3.11
conda activate edm

# Install all dependencies
python install_dependencies.py
```

### Alternative Installation Methods

#### Method 1: Shell script
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

#### Method 2: Conda environment file
```bash
conda env create -f environment.yml
conda activate edm
```

#### Method 3: Manual installation
```bash
# Install OpenBabel via conda
conda install -c conda-forge openbabel

# Install Python packages
pip install numpy>=1.24.0 PyYAML>=6.0.0 tqdm>=4.60.0 rdkit>=2023.0.0

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## How to install openbabel
1. Activate your `conda` environment (edm in this example)
```bash
$ conda activate edm 
``` 
2. Choose an installation path (in this example it is a folder named `openbabel` inside the home path)
```bash
$ (edm) mkdir openbabel
$ (edm) cd openbabel
```
3. Download openbabel  from [here](https://github.com/openbabel/openbabel/archive/refs/tags/openbabel-3-1-1.tar.gz) 
4. Run the following instructions. **Note** that you might need to install additional system packages depending on your OS. Please check the official [requirement list](https://openbabel.org/docs/Installation/install.html#requirements)
```bash
$ (edm) tar -zxvf openbabel-3-1-1.tar.gz
$ (edm) mkdir build
$ (edm) cd build
$ (edm) OB_INST_DIR=$(dirname "$(pwd)")
$ (edm) cmake ../openbabel-openbabel-3-1-1  -DCMAKE_INSTALL_PREFIX="${OB_INST_DIR}"/openbabel-install-3-1-1 -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON
$ (edm) make -j16 # 16 is the number of cpus (lower or increase it depending on your hardware)
$ (edm) make install
$ (edm) export PYTHONPATH=${OB_INST_DIR}/openbabel-install-3-1-1/lib/python3.11/site-packages:$PYTHONPATH
```

## Generate molecules
```bash
$ (edm) python sample_mols.py --checkpoint checkpoints/model_updates_738999.ckpt --n-molecules 10 --n-atoms 40  --output-dir raw_mols --device cuda
```

## Generate molecules with energy guidance
```bash
$ (edm) python sample_mols.py --checkpoint checkpoints/model_updates_738999.ckpt --n-molecules 10 --n-atoms 40  --output-dir raw_mols --beta 10 --start_t 200 --device cuda --x0-guidance
```

## Generate molecules with energy guidance into protein pockets
```bash
$ (edm) python conditional_sample_mols.py --checkpoint checkpoints/conditional_model_updates_487_epochs.ckpt  --protein dataset/pdbbind/5zcu  --n-molecules 10 --n-atoms 30  --output-dir raw_mols --beta=10 --guidance-time=200 --device cuda --x0-guidance
```

## Convert point clouds into SDF files
```bash
$ (edm) python src/guidance_plugins/utils/cloud2mol.py raw_mols/10_40_000.npy molecules_10_atoms_40.sdf
```
