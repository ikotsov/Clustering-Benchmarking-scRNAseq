import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.stats import gaussian_kde

from .constants import DEFAULT_SEED, DEFAULT_STRATIFIED_SAMPLES
from .types import StratifiedHVGSampleSet


def build_stratified_hvg_samples(
    gene_sampling_features: pd.DataFrame,
    hvg_genes: list[str],
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> StratifiedHVGSampleSet:
    """Build observed HVGs and stratified random samples."""
    gene_sampling_probabilities = _estimate_gene_probabilities(
        gene_sampling_features, hvg_genes)
    rng = np.random.default_rng(seed)
    all_genes = gene_sampling_features.index.to_numpy()
    control_samples: list[list[str]] = []

    for _ in range(1, n_samples + 1):
        # matched random sets (same size, similar feature distribution) used as a null/control reference.
        sampled_hvg_control_genes = rng.choice(
            all_genes,
            size=len(hvg_genes),
            replace=False,
            p=gene_sampling_probabilities,
            shuffle=False,
        )
        control_samples.append(sampled_hvg_control_genes.tolist())

    return StratifiedHVGSampleSet(
        observed_hvg_genes=hvg_genes,
        control_samples=control_samples,
    )


def _estimate_gene_probabilities(gene_statistics: pd.DataFrame, observed_genes: list[str]) -> np.ndarray:
    """Estimate KDE-based sampling probability for gene universe."""
    observed_hvg_features = gene_statistics.loc[observed_genes]
    all_gene_features = gene_statistics

    # fallback, fair chance for all genes.
    uniform_distribution = np.full(
        len(gene_statistics), 1.0 / len(gene_statistics))

    try:
        kde = gaussian_kde(observed_hvg_features.T.to_numpy())
        probabilities = kde.evaluate(all_gene_features.T.to_numpy())
    except (ValueError, LinAlgError, FloatingPointError):
        return uniform_distribution

    probabilities = np.clip(probabilities, a_min=0.0, a_max=None)
    total = probabilities.sum()

    if total <= 0:
        return uniform_distribution

    return probabilities / total
