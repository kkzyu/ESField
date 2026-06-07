"""Centralized, optimized default parameters for ESField pipeline.

These values are calibrated for:
  - Single GPU: RTX 4090D 24GB (or A100 40GB)
  - Training data: ~500-1000 PDBbind pockets
  - Generation: ~20-30 test pockets
"""

# --- Compatibility Potential Training ---
TRAINING = {
    "hidden_dim": 128,       # MLP hidden dimension; 256 for larger datasets
    "num_layers": 4,         # Residual MLP blocks
    "atom_embed_dim": 32,    # Atom type embedding
    "site_embed_dim": 32,    # Site type embedding
    "rbf_bins": 16,          # Gaussian RBF bins for distance encoding
    "cutoff": 6.0,           # Distance cutoff (Angstrom)
    "energy_clip": 5.0,      # tanh clip range for energy output
    "epochs": 150,           # Sufficient; loss plateaus ~epoch 100
    "batch_size": 2048,      # Safe for 24GB; 4096 for A100 40GB
    "lr": 1e-4,              # AdamW learning rate
    "weight_decay": 1e-5,    # AdamW weight decay
    "grad_clip": 1.0,        # Gradient norm clipping
    "loss": "margin",        # "margin" or "logistic"
    "margin_pos": -1.0,      # Positive energy target (below this)
    "margin_neg": 1.0,       # Negative energy target (above this)
    "save_every": 20,        # Checkpoint frequency (epochs)
    "seed": 20260511,
}

# --- Training Pair Construction ---
PAIRS = {
    "positive_padding": 0.5,      # Extra distance beyond site radius
    "sigma_scale": 1.5,           # Gaussian sigma = sigma_scale × site_radius
    "max_positive_sites_per_atom": 2,  # Max positive sites per ligand atom
    "negative_ratio": 3,          # Negatives per positive
    "displacement_threshold": 4.0,     # Water displacement search radius (Angstrom)
    "hew_max_positive_per_site": 3,    # Max aspirational positive pairs per HEW
    "sw_max_positive_per_site": 2,     # Max aspirational positive pairs per SW
    "hew_strength_mult": 0.85,         # Label strength for HEW displacement pairs
    "sw_strength_mult": 0.50,          # Label strength for SW displacement pairs
}

# --- Guided Generation ---
GENERATION = {
    "timesteps": 40,             # ODE integration steps (was 50)
    "guidance_start": 0.4,       # Fraction of flow when guidance begins (was 0.3)
    "guidance_end": 0.85,        # Fraction of flow when guidance ends (was 0.90)
    "grad_clip": 1.0,            # Per-atom gradient norm clip
    "gen_batch_size": 5,         # Molecules per GPU batch (safe for 24GB)
    "lambda_sweep": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    "num_samples_per_condition": 20,
}

# --- Site Detection ---
SITE_DETECTION = {
    "pocket_radius": 10.0,        # Crystal water search radius around pocket center
    "water_radius": 1.4,          # Water site radius (Angstrom)
    "protein_clash_distance": 1.6,     # Minimum water-protein distance
    "hbond_distance": 3.5,        # H-bond cutoff (Angstrom)
    "hydrophobic_distance": 4.0,  # Hydrophobic contact cutoff (Angstrom)
    "stable_hbond_min": 2,        # Min H-bonds for stable water
    "high_energy_hbond_max": 1,   # Max H-bonds for high-energy water
    "high_energy_hydrophobic_min": 3,  # Min hydrophobic contacts for HEW
    "fpocket_max_sites": 10,      # Max fpocket-derived sites per pocket
    "water_max_sites": 15,        # Max crystal-water-derived sites per pocket
    "merge_max_sites": 20,        # Max merged sites per pocket
    "merge_distance": 1.0,        # Spatial clustering distance (Angstrom)
}

# --- Evaluation ---
EVALUATION = {
    "sigma_scale": 1.5,          # Gaussian proximity sigma = sigma_scale × radius
    "qed_threshold": 0.4,        # Minimum QED for "globally plausible"
    "sa_threshold": 5.0,         # Maximum SA score for "globally plausible"
    "vina_percentile": 30.0,     # Top percentile Vina for "globally plausible"
    "penalty_weight": 1.0,       # Q-POSU quality penalty weight
}

# --- Experiment Plan ---
EXPERIMENT = {
    "n_train_pockets": 500,      # Target PDBbind training pockets
    "n_test_pockets": 20,        # Target held-out test pockets
    "n_ablation_pockets": 5,     # Pockets for Step 2 (site meaningfulness)
    "n_lambda_sweep_pockets": 10, # Pockets for Step 4 (quality constraint)
    "n_samples_per_condition": 20,
    "total_generations": None,   # Computed below
}

# Compute total GPU generations needed (updated 2026-05-19)
# Step 1: 20 test pockets × 1 condition (baseline λ=0) — fresh generation needed
# Step 2: 5 ablation pockets × 3 site map conditions (correct/random/shuffled) × λ=1.0
# Step 3: 20 pockets × 3 conditions (λ=0.5, λ=1.0, random) — baseline reused from Step 1
# Step 4: 10 pockets × 8 λ values (sweep 0.0–2.0)
# Dedup: Step 3 random reuses Step 2 random for 5 overlapping pockets (symlink)
# Dedup: Step 3 baseline reuses Step 1 baseline (symlink)

_test = EXPERIMENT["n_test_pockets"]
_step1 = _test * EXPERIMENT["n_samples_per_condition"]  # 20 × 20 = 400
_step2 = EXPERIMENT["n_ablation_pockets"] * 3 * EXPERIMENT["n_samples_per_condition"]  # 5 × 3 × 20 = 300
_step3_new = _test * 2 * EXPERIMENT["n_samples_per_condition"]  # λ=0.5, λ=1.0 (20 × 2 × 20 = 800)
_step3_random = (_test - EXPERIMENT["n_ablation_pockets"]) * EXPERIMENT["n_samples_per_condition"]  # 15 × 20 = 300
_step3 = _step3_new + _step3_random  # 1100 new + 25 reused via symlinks
_step4 = EXPERIMENT["n_lambda_sweep_pockets"] * len(GENERATION["lambda_sweep"]) * EXPERIMENT["n_samples_per_condition"]  # 10 × 8 × 20 = 1600
EXPERIMENT["total_generations"] = _step1 + _step2 + _step3 + _step4  # 400 + 300 + 1100 + 1600 = 3400
EXPERIMENT["total_runs"] = (_step1 + _step2 + _step3 + _step4) // EXPERIMENT["n_samples_per_condition"]  # 170 runs
EXPERIMENT["estimated_gpu_hours"] = EXPERIMENT["total_generations"] * 4.5 / 3600  # ~4.25h A100

print(f"[ESField Config] Total GPU generations: {EXPERIMENT['total_generations']}")
print(f"[ESField Config] Total runs: {EXPERIMENT['total_runs']}")
print(f"[ESField Config] Estimated GPU hours: {EXPERIMENT['estimated_gpu_hours']:.1f}h (A100)")
print(f"[ESField Config] Step 1 (baseline): {_step1} gens, Step 2 (site meaning): {_step2} gens")
print(f"[ESField Config] Step 3 (improvement): {_step3} gens, Step 4 (quality): {_step4} gens")
