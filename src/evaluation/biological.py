from dataclasses import dataclass
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
import scanpy as sc
from scipy.stats import gaussian_kde


EnrichmentSetName = Literal["de_markers"]


def build_dataset_biological_enrichment_sets(
    expression_data: pd.DataFrame,
    cell_type_labels: pd.Series,
    enrichment_sets: list[EnrichmentSetName],
) -> dict[EnrichmentSetName, dict[str, set[str]]]:
    """Build all requested biological enrichment sets for one dataset."""
    return {
        enrichment_set: build_biological_enrichment_set(
            expression_data,
            cell_type_labels,
            enrichment_set,
        )
        for enrichment_set in enrichment_sets
    }


def build_biological_enrichment_set(
    expression_data: pd.DataFrame,
    cell_type_labels: pd.Series,
    enrichment_set: EnrichmentSetName,
) -> dict[str, set[str]]:
    """Build one supported enrichment set for a dataset."""
    if enrichment_set == "de_markers":
        return build_de_marker_enrichment_set(expression_data, cell_type_labels)

    raise ValueError(f"Unsupported enrichment set: {enrichment_set}")


# Configuration for DEG-based enrichment sets.
DEFAULT_DEG_PVAL_CUTOFF = 0.01
DEFAULT_DEG_POSITIVE_LOGFC = 1.0
DEFAULT_DEG_NEGATIVE_LOGFC = -1.0
DEFAULT_DEG_METHOD = "wilcoxon"


def build_de_marker_enrichment_set(
    expression_data: pd.DataFrame,
    cell_type_labels: pd.Series,
) -> dict[str, set[str]]:
    """Build DEG-based enrichment sets per cell type.

    Positive markers (logFC >= threshold) are combined with negative markers
    (logFC <= threshold) after removing genes that are positive in any cell type.
    """
    common_cells = expression_data.index.intersection(cell_type_labels.index)
    if len(common_cells) == 0:
        raise ValueError(
            "No overlapping cells between expression data and labels.")

    matrix = expression_data.loc[common_cells]
    labels = cell_type_labels.loc[common_cells].astype(str)

    adata = ad.AnnData(matrix)
    adata.obs["cell_type"] = labels

    sc.tl.rank_genes_groups(
        adata,
        groupby="cell_type",
        method=DEFAULT_DEG_METHOD,
    )
    de_results = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        pval_cutoff=DEFAULT_DEG_PVAL_CUTOFF,
    )

    if de_results.empty:
        return {str(cell_type): set() for cell_type in labels.unique()}

    cell_types = de_results["group"].astype(str).unique()

    positive_markers_by_cell_type = {
        cell_type: {
            str(gene)
            for gene in pd.Series(
                de_results.loc[
                    (de_results["group"].astype(str) == cell_type)
                    & (de_results["logfoldchanges"] >= DEFAULT_DEG_POSITIVE_LOGFC),
                    "names",
                ]
            ).tolist()
            if pd.notna(gene)
        }
        for cell_type in cell_types
    }

    negative_markers_by_cell_type = {
        cell_type: {
            str(gene)
            for gene in pd.Series(
                de_results.loc[
                    (de_results["group"].astype(str) == cell_type)
                    & (de_results["logfoldchanges"] <= DEFAULT_DEG_NEGATIVE_LOGFC),
                    "names",
                ]
            ).tolist()
            if pd.notna(gene)
        }
        for cell_type in cell_types
    }

    all_positive = set().union(*positive_markers_by_cell_type.values()
                               ) if positive_markers_by_cell_type else set()

    cleaned_negative_markers_by_cell_type = {
        cell_type: genes - all_positive
        for cell_type, genes in negative_markers_by_cell_type.items()
    }

    return {
        cell_type: cleaned_negative_markers_by_cell_type.get(cell_type, set())
        | positive_markers_by_cell_type.get(cell_type, set())
        for cell_type in cell_types
    }


DEFAULT_STRATIFIED_SAMPLES = 100
DEFAULT_SEED = 3


@dataclass(frozen=True)
class StratifiedHVGSampleSet:
    """Container for observed HVGs and matched random control samples."""

    observed_hvg_genes: list[str]
    control_samples: list[list[str]]


def build_dataset_stratified_hvgs(
    gene_statistics: pd.DataFrame,
    hvg_genes: list[str],
    norm_method: str,
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, StratifiedHVGSampleSet]:
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
) -> StratifiedHVGSampleSet:
    """Build observed HVGs and stratified random samples."""
    gene_sampling_probabilities = _estimate_gene_probabilities(
        gene_statistics, hvg_genes)
    rng = np.random.default_rng(seed)
    all_genes = gene_statistics.index.to_numpy()
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


def compute_gene_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute gene-level statistics for stratifying HVGs."""
    stats = pd.DataFrame(index=data.columns)
    # number of cells where the gene is expressed.
    stats["num_expressed_cells"] = (data > 0).sum(axis=0).astype(int)
    # mean expression of the gene across all cells.
    stats["mean_expression"] = data.mean(axis=0)
    stats.index.name = "gene"
    return stats
