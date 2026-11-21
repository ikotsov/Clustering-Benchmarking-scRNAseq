import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.stats import gaussian_kde

from .constants import DEFAULT_SEED, DEFAULT_STRATIFIED_SAMPLES


def compute_gene_sampling_features(counts: pd.DataFrame) -> pd.DataFrame:
    """Describe each gene (column) of a raw counts matrix by its expression level.

    - ``log_expressing_cells``: log(1 + number of cells with a count > 0)
    - ``log_mean_count``: log(1 + mean raw count over all cells)

    Raw counts, not normalized values, so that "expressed" means a count > 0 for every
    normalization (a Pearson residual is negative for most zeros). Log scale, so the KDE
    in `build_matched_gene_samples` is not dominated by a few very highly expressed genes.
    """
    features = pd.DataFrame({
        "log_expressing_cells": np.log1p((counts > 0).sum(axis=0).astype(float)),
        "log_mean_count": np.log1p(counts.mean(axis=0).astype(float)),
    }, index=counts.columns)
    features.index.name = "gene"
    return features


def build_matched_gene_samples(
    gene_sampling_features: pd.DataFrame,
    observed_genes: list[str],
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[list[str]]:
    """Draw random gene sets that are the size of `observed_genes` and expressed like them.

    A 2D Gaussian KDE is fitted to the observed genes' features; every gene of the universe
    (the index of `gene_sampling_features`) gets a sampling probability proportional to the
    KDE at its features; each sample draws len(observed_genes) genes without replacement.
    The universe must contain more genes than the observed set, or every sample is the
    whole universe.
    """
    if len(observed_genes) >= len(gene_sampling_features):
        raise ValueError(
            f"Cannot draw random sets of {len(observed_genes)} genes from a universe of "
            f"{len(gene_sampling_features)} genes: every sample would be the whole universe."
        )
    probabilities = _estimate_gene_probabilities(gene_sampling_features, observed_genes)
    rng = np.random.default_rng(seed)
    universe = gene_sampling_features.index.to_numpy()

    return [
        rng.choice(universe, size=len(observed_genes), replace=False, p=probabilities).tolist()
        for _ in range(n_samples)
    ]


def _estimate_gene_probabilities(gene_statistics: pd.DataFrame, observed_genes: list[str]) -> np.ndarray:
    """Estimate KDE-based sampling probability for gene universe."""
    observed_features = gene_statistics.loc[observed_genes]

    # fallback, fair chance for all genes (e.g. too few observed genes to fit a 2D KDE).
    uniform_distribution = np.full(len(gene_statistics), 1.0 / len(gene_statistics))

    try:
        kde = gaussian_kde(observed_features.T.to_numpy())
        probabilities = kde.evaluate(gene_statistics.T.to_numpy())
    except (ValueError, LinAlgError, FloatingPointError):
        return uniform_distribution

    probabilities = np.clip(probabilities, a_min=0.0, a_max=None)
    total = probabilities.sum()

    if not np.isfinite(total) or total <= 0:
        return uniform_distribution

    return probabilities / total
