import logging

import pandas as pd
from .types import PreprocessingConfig
from .filters import filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import select_hvgs, normalize_with_log_cpm, normalize_with_pearson
from src.types import NormMethod, Species


logger = logging.getLogger(__name__)


def preprocess_data(
    raw_data: pd.DataFrame,
    norm_method: NormMethod = "pearson",
    species: Species = "human",
    preprocessing_config: PreprocessingConfig = PreprocessingConfig(),
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
    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Processed data (HVG space) ready for optional downstream PCA and selected HVG names
    """
    # Filter first
    clean_data = run_filtering_pipeline(
        raw_data, config=preprocessing_config, species=species)

    # Then select HVGs
    hvg_genes = select_hvgs(clean_data, norm_method)
    hvg_data = clean_data.loc[:, hvg_genes]

    # Then normalize
    if norm_method == "log_cpm":
        logger.debug("Normalization (LogCPM)...")
        normalized_data = normalize_with_log_cpm(hvg_data)
    elif norm_method == "pearson":
        logger.debug("Normalization (Pearson Residuals)...")
        normalized_data = normalize_with_pearson(hvg_data)
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    return normalized_data, hvg_genes


def run_filtering_pipeline(raw_data: pd.DataFrame, config: PreprocessingConfig, species: Species = "human") -> pd.DataFrame:
    """
    Runs the full filtering pipeline.
    """
    logger.info("Filtering...")
    logger.info("Input: %s cells x %s genes",
                raw_data.shape[0], raw_data.shape[1])

    data = filter_low_magnitude_genes(
        raw_data, min_count=config.gene_magnitude_threshold)
    data = filter_high_apoptosis_cells(
        data, species=species, threshold=config.apoptosis_threshold)
    data = filter_high_rrna_cells(
        data, species=species, threshold=config.rrna_threshold)
    data = filter_high_mito_cells(data, threshold=config.mito_threshold)

    logger.info("Output: %s cells x %s genes", data.shape[0], data.shape[1])
    return data
