"""Molecular diversity metrics for v7 ablation study.

Provides:
  - compute_vendi_score:     Vendi score (entropy of normalized similarity kernel)
  - compute_pairwise_tanimoto: mean pairwise Tanimoto similarity
  - compute_diversity_metrics: combined diversity report

Reference:
  Vendi Score: Friedman & Dieng (2023), "The Vendi Score: A Diversity
  Evaluation Metric for Machine Learning"
  https://arxiv.org/abs/2210.02410

  The Vendi score is the exponential of the entropy of the eigenvalues
  of the normalized similarity kernel K_ij = S_ij / sqrt(S_ii * S_jj),
  where S_ij = exp(-d(x_i, x_j) / sigma) with d = 1 - Tanimoto.

  Higher Vendi score = more diverse set of molecules.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

# Suppress RDKit warnings during bulk computation
RDLogger.logger().setLevel(RDLogger.ERROR)


# ---------------------------------------------------------------------------
# Morgan fingerprint generation
# ---------------------------------------------------------------------------

def _mols_to_fingerprints(
    molecules: list,
    radius: int = 2,
    nbits: int = 2048,
) -> list[DataStructs.ExplicitBitVect]:
    """Convert RDKit molecules to Morgan (ECFP) fingerprints.

    Args:
        molecules: list of RDKit Mol objects
        radius: Morgan fingerprint radius (default 2 = ECFP4)
        nbits: fingerprint bit length

    Returns:
        List of ExplicitBitVect fingerprints (same order as input).
        Invalid molecules are skipped (result may be shorter than input).
    """
    fps = []
    for mol in molecules:
        if mol is None:
            continue
        try:
            # Sanitize if needed (some DrugFlow outputs have valence issues)
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass  # fingerprint may still work on unsanitized mol
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(fp)
        except Exception:
            continue
    return fps


# ---------------------------------------------------------------------------
# Pairwise Tanimoto similarity
# ---------------------------------------------------------------------------

def compute_pairwise_tanimoto(
    molecules: list,
    radius: int = 2,
    nbits: int = 2048,
) -> dict:
    """Compute mean pairwise Tanimoto similarity among molecules.

    Args:
        molecules: list of RDKit Mol objects
        radius: Morgan fingerprint radius
        nbits: fingerprint bit length

    Returns:
        dict with:
          - "mean_tanimoto": mean of all pairwise similarities
          - "std_tanimoto": std of pairwise similarities
          - "min_tanimoto": minimum pairwise similarity
          - "max_tanimoto": maximum pairwise similarity
          - "n_pairs": number of valid pairs
          - "n_mols": number of valid molecules
    """
    fps = _mols_to_fingerprints(molecules, radius=radius, nbits=nbits)
    n = len(fps)

    if n < 2:
        return {
            "mean_tanimoto": 0.0,
            "std_tanimoto": 0.0,
            "min_tanimoto": 0.0,
            "max_tanimoto": 0.0,
            "n_pairs": 0,
            "n_mols": n,
        }

    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    sims = np.array(similarities)
    return {
        "mean_tanimoto": float(np.mean(sims)),
        "std_tanimoto": float(np.std(sims)),
        "min_tanimoto": float(np.min(sims)),
        "max_tanimoto": float(np.max(sims)),
        "n_pairs": len(sims),
        "n_mols": n,
    }


# ---------------------------------------------------------------------------
# Vendi Score
# ---------------------------------------------------------------------------

def compute_vendi_score(
    molecules: list,
    radius: int = 2,
    nbits: int = 2048,
    sigma: float | None = None,
) -> dict:
    """Compute Vendi diversity score for a set of molecules.

    The Vendi score VS = exp(-Σ_i λ_i log λ_i) where λ_i are the
    eigenvalues of the normalized similarity kernel K / Tr(K).

    Uses 1 - Tanimoto as the distance metric.

    Args:
        molecules: list of RDKit Mol objects
        radius: Morgan fingerprint radius
        nbits: fingerprint bit length
        sigma: kernel bandwidth. If None, auto-tuned as the median
               of pairwise distances.

    Returns:
        dict with:
          - "vendi_score": float (higher = more diverse)
          - "effective_number": int (VS ≈ number of distinct clusters)
          - "entropy": float (entropy of eigenvalue distribution)
          - "n_mols": number of valid molecules used
    """
    fps = _mols_to_fingerprints(molecules, radius=radius, nbits=nbits)
    n = len(fps)

    if n < 2:
        return {
            "vendi_score": 1.0 if n == 1 else 0.0,
            "effective_number": n,
            "entropy": 0.0,
            "n_mols": n,
        }

    # Compute pairwise distance matrix (1 - Tanimoto)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            d = 1.0 - sim
            D[i, j] = d
            D[j, i] = d

    # Auto-tune sigma if not provided
    if sigma is None:
        # Use median of upper-triangular distances
        triu_idx = np.triu_indices(n, k=1)
        sigma = float(np.median(D[triu_idx]))
        if sigma < 1e-8:
            sigma = 0.1  # fallback for identical molecules

    # Build similarity kernel: K_ij = exp(-D_ij / sigma)
    K = np.exp(-D / sigma)

    # Normalize: K_norm = K / Tr(K) so that eigenvalues sum to 1
    trace_K = np.trace(K)
    if trace_K < 1e-12:
        K_norm = np.eye(n) / n
    else:
        K_norm = K / trace_K

    # Eigen-decomposition
    eigenvalues = np.linalg.eigvalsh(K_norm)
    # Ensure numerical stability: clip negative eigenvalues to 0
    eigenvalues = np.clip(eigenvalues, 0, None)
    # Re-normalize to sum to 1
    eig_sum = eigenvalues.sum()
    if eig_sum > 1e-12:
        eigenvalues = eigenvalues / eig_sum

    # Entropy: -Σ λ_i log λ_i (use natural log)
    entropy = 0.0
    for lam in eigenvalues:
        if lam > 1e-12:
            entropy -= lam * np.log(lam)

    # Vendi score = exp(entropy)
    vendi_score = float(np.exp(entropy))
    effective_number = int(np.round(vendi_score))

    return {
        "vendi_score": vendi_score,
        "effective_number": effective_number,
        "entropy": float(entropy),
        "n_mols": n,
    }


# ---------------------------------------------------------------------------
# Combined diversity report
# ---------------------------------------------------------------------------

def compute_diversity_metrics(
    molecules: list,
    radius: int = 2,
    nbits: int = 2048,
    vendi_sigma: float | None = None,
) -> dict:
    """Compute all diversity metrics for a set of molecules.

    Args:
        molecules: list of RDKit Mol objects
        radius: Morgan fingerprint radius
        nbits: fingerprint bit length
        vendi_sigma: kernel bandwidth for Vendi score (auto if None)

    Returns:
        Combined dict with Tanimoto stats and Vendi score.
    """
    tanimoto = compute_pairwise_tanimoto(molecules, radius=radius, nbits=nbits)
    vendi = compute_vendi_score(molecules, radius=radius, nbits=nbits, sigma=vendi_sigma)

    return {
        "n_mols": tanimoto["n_mols"],
        "mean_pairwise_tanimoto": tanimoto["mean_tanimoto"],
        "std_pairwise_tanimoto": tanimoto["std_tanimoto"],
        "vendi_score": vendi["vendi_score"],
        "vendi_effective_n": vendi["effective_number"],
        "vendi_entropy": vendi["entropy"],
    }
