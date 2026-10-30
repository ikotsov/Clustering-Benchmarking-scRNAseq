import logging

import pandas as pd
from .types import PreprocessingConfig
from .filters import (
    filter_genes_by_max_count,
    filter_rrna_genes,
    high_fraction_mask,
    low_count_depth_mask,
    low_gene_count_mask,
)
from .gene_sets import mito_genes
from .transforms import (
    select_hvgs_seurat_v3,
    select_hvgs_pearson,
    normalize_with_log_cpm,
    normalize_with_pearson,
)
from src.types import NormMethod, Species


logger = logging.getLogger(__name__)


def preprocess_data(
    raw_data: pd.DataFrame,
    norm_method: NormMethod = "pearson",
    species: Species = "human",
    preprocessing_config: PreprocessingConfig = PreprocessingConfig(),
    annotation: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Runs filtering and normalization.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw gene expression data (cells x genes)
    norm_method : NormMethod, default="pearson"
        Normalization method to use ("log_cpm" or "pearson")
    species : Species, default="human"
        Species for filtering (affects mitochondrial, ribosomal, apoptosis genes)
    annotation : pd.DataFrame, optional
        Gene annotation table of the species used to build the QC gene sets
        (defaults to the project's annotation file for the species)
    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Processed data (HVG space) ready for optional downstream PCA and selected HVG names
    """
    # Filter first
    clean_data = run_filtering_pipeline(
        raw_data, config=preprocessing_config, species=species, annotation=annotation)

    # Both HVG flavors score genes on raw counts (clean_data). The methods
    # differ in whether we keep only the HVGs before or after normalizing.
    if norm_method == "log_cpm":
        logger.debug("Normalization (LogCPM)...")
        hvg_genes = select_hvgs_seurat_v3(clean_data)
        # Normalize on all genes so library sizes reflect sequencing depth,
        # then keep only the HVG columns.
        log_cpm_all_genes = normalize_with_log_cpm(clean_data)
        normalized_data = log_cpm_all_genes.loc[:, hvg_genes]
    elif norm_method == "pearson":
        logger.debug("Normalization (Pearson Residuals)...")
        hvg_genes = select_hvgs_pearson(clean_data)
        # Keep only the HVG columns first, then compute residuals on them,
        # as in Scanpy's Pearson residuals workflow.
        hvg_counts = clean_data.loc[:, hvg_genes]
        normalized_data = normalize_with_pearson(hvg_counts)
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    return normalized_data, hvg_genes


def run_filtering_pipeline(
    raw_data: pd.DataFrame,
    config: PreprocessingConfig,
    species: Species = "human",
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Runs the full filtering pipeline.
    """
    logger.info("Filtering...")
    logger.info("Input: %s cells x %s genes",
                raw_data.shape[0], raw_data.shape[1])

    low_counts = low_count_depth_mask(raw_data, config.count_threshold)
    low_genes = low_gene_count_mask(raw_data, config.gene_threshold)
    high_mito = high_fraction_mask(
        raw_data, mito_genes(species, annotation), config.mito_threshold, "mitochondrial")
    logger.info("Cells with count depth < %s: %s",
                config.count_threshold, int(low_counts.sum()))
    logger.info("Cells with genes detected < %s: %s",
                config.gene_threshold, int(low_genes.sum()))
    logger.info("Cells with high mitochondrial fraction: %s",
                int(high_mito.sum()))

    failed_qc = low_counts | low_genes | high_mito
    logger.info("Dropped %s cells failing cell QC", int(failed_qc.sum()))

    data = raw_data.loc[~failed_qc]

    n_genes = data.shape[1]
    data = filter_rrna_genes(data, species=species, annotation=annotation)
    logger.info("Dropped %s rRNA genes", n_genes - data.shape[1])

    data = filter_genes_by_max_count(
        data, min_count=config.min_gene_max_count)

    logger.info("Output: %s cells x %s genes", data.shape[0], data.shape[1])
    return data
