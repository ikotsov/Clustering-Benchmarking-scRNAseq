import logging

import pandas as pd
from .types import PreprocessingConfig
from .filters import filter_genes_by_max_count, high_fraction_mask
from .gene_sets import apoptosis_genes, mito_genes, rrna_genes
from .transforms import select_hvgs, normalize_with_log_cpm, normalize_with_pearson
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

    # Then select HVGs (both flavors expect raw counts)
    hvg_genes = select_hvgs(clean_data, norm_method)

    # Then normalize
    if norm_method == "log_cpm":
        logger.debug("Normalization (LogCPM)...")
        # Normalize on all genes first so library sizes reflect sequencing depth,
        # then keep only the HVG columns.
        log_cpm_all_genes = normalize_with_log_cpm(clean_data)
        normalized_data = log_cpm_all_genes.loc[:, hvg_genes]
    elif norm_method == "pearson":
        logger.debug("Normalization (Pearson Residuals)...")
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

    # Compute every cell QC mask on the raw matrix so each count reflects that
    # filter alone, independent of the order the filters are listed in.
    qc_failures = {
        "apoptosis": high_fraction_mask(
            raw_data, apoptosis_genes(species, annotation), config.apoptosis_threshold),
        "rRNA": high_fraction_mask(
            raw_data, rrna_genes(species, annotation), config.rrna_threshold),
        "mitochondrial": high_fraction_mask(
            raw_data, mito_genes(species, annotation), config.mito_threshold),
    }
    for name, mask in qc_failures.items():
        logger.info("High %s: %s cells", name, int(mask.sum()))

    failed_any = pd.concat(qc_failures.values(), axis=1).any(axis=1)
    logger.info("Dropped %s cells failing at least one QC filter",
                int(failed_any.sum()))

    data = raw_data.loc[~failed_any]
    data = filter_genes_by_max_count(
        data, min_count=config.min_gene_max_count)

    logger.info("Output: %s cells x %s genes", data.shape[0], data.shape[1])
    return data
