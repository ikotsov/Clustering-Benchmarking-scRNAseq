import pandas as pd
from .types import PreprocessingConfig
from .filters import filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform, normalize_data_with_pearson
from .dimensionality import apply_pca
from src.constants import PCA_VARIANCE_RATIO
from src.types import NormMethod, Species


def preprocess_data(
    raw_data: pd.DataFrame,
    norm_method: NormMethod = "pearson",
    species: Species = "human",
    pca_variance_ratio: float = PCA_VARIANCE_RATIO,
    preprocessing_config: PreprocessingConfig = PreprocessingConfig(),
    with_pca: bool = True,
) -> pd.DataFrame:
    """
    Runs filtering, normalization, and PCA dimensionality reduction.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw gene expression data (cells × genes)
    norm_method : NormMethod, default="pearson"
        Normalization method to use ("log_cpm" or "pearson")
    species : Species, default="human"
        Species for filtering (affects mitochondrial, ribosomal, apoptosis genes)
    pca_variance_ratio : float, default=PCA_VARIANCE_RATIO
        Fraction of variance to preserve when PCA is applied

    Returns
    -------
    pd.DataFrame
        Processed data ready for clustering
    """
    # Always filter first
    clean_data = filter_data(
        raw_data, config=preprocessing_config, species=species)

    # Selective normalization
    if norm_method == "log_cpm":
        print()
        print("Normalization (LogCPM)...")
        normalized_data = normalize_data_with_log_cpm(clean_data)

    elif norm_method == "pearson":
        print()
        print("Normalization (Pearson Residuals)...")
        normalized_data = normalize_data_with_pearson(clean_data)

    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    if not with_pca:
        print()
        print("Skipping dimensionality reduction (PCA)...")
        return normalized_data

    # Apply PCA
    print()
    print("Dimensionality reduction (PCA)...")
    return apply_pca(normalized_data, variance_ratio=pca_variance_ratio)


def normalize_data_with_log_cpm(filtered_data: pd.DataFrame) -> pd.DataFrame:

    data = normalize_by_library_size(filtered_data)
    data = log_transform(data)

    return data


def filter_data(raw_data: pd.DataFrame, config: PreprocessingConfig, species: Species = "human") -> pd.DataFrame:
    """
    Runs the full filtering pipeline.
    """
    print(f"Filtering...")
    print(f"  Input: {raw_data.shape[0]} cells × {raw_data.shape[1]} genes")

    data = filter_low_magnitude_genes(
        raw_data, min_count=config.gene_magnitude_threshold)
    data = filter_high_apoptosis_cells(
        data, species=species, threshold=config.apoptosis_threshold)
    data = filter_high_rrna_cells(
        data, species=species, threshold=config.rrna_threshold)
    data = filter_high_mito_cells(data, threshold=config.mito_threshold)

    print(f"  Output: {data.shape[0]} cells × {data.shape[1]} genes")
    return data
