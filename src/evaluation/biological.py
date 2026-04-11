import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.stats import gaussian_kde

DEFAULT_STRATIFIED_SAMPLES = 100
DEFAULT_SEED = 3


def build_dataset_stratified_hvgs(
    gene_statistics: pd.DataFrame,
    hvg_genes: list[str],
    norm_method: str,
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, list[str]]]:
    """Build stratified HVGs grouped by normalization method."""
    return {
        norm_method: build_stratified_hvg_samples(
            gene_statistics=gene_statistics,
            hvg_genes=hvg_genes,
            n_samples=n_samples,
            seed=seed,
        )
    }


def build_stratified_hvg_samples(
    gene_statistics: pd.DataFrame,
    hvg_genes: list[str],
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Build observed HVGs and stratified random samples."""
    gene_sampling_probabilities = _estimate_gene_probabilities(
        gene_statistics, hvg_genes)
    rng = np.random.default_rng(seed)
    stratified_hvg_samples = {"observed_hvg_genes": hvg_genes}
    all_genes = gene_statistics.index.to_numpy()

    for sample_idx in range(1, n_samples + 1):
        # matched random sets (same size, similar feature distribution) used as a null/control reference.
        sampled_hvg_control_genes = rng.choice(
            all_genes,
            size=len(hvg_genes),
            replace=False,
            p=gene_sampling_probabilities,
            shuffle=False,
        )
        stratified_hvg_samples[f"sample_{sample_idx}"] = sampled_hvg_control_genes.tolist(
        )

    return stratified_hvg_samples


STRATIFICATION_FEATURE_COLUMNS = ("num_expressed_cells", "mean_expression")


def _estimate_gene_probabilities(gene_statistics: pd.DataFrame, observed_genes: list[str]) -> np.ndarray:
    """Estimate KDE-based sampling probability for gene universe."""
    observed_hvg_features = gene_statistics.loc[observed_genes,
                                                list(STRATIFICATION_FEATURE_COLUMNS)]
    all_gene_features = gene_statistics.loc[:, list(
        STRATIFICATION_FEATURE_COLUMNS)]

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


def compute_gene_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute gene-level statistics for stratifying HVGs."""
    stats = pd.DataFrame(index=data.columns)
    # number of cells where the gene is expressed.
    stats["num_expressed_cells"] = (data > 0).sum(axis=0).astype(int)
    # mean expression of the gene across all cells.
    stats["mean_expression"] = data.mean(axis=0)
    stats.index.name = "gene"
    return stats
