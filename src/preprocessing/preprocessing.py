import pandas as pd
from .types import NormMethod
from .filters import Species, filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform, normalize_data_with_pearson
from .dimensionality import apply_pca
from src.constants import N_PCA_COMPONENTS


def preprocess_data(
    raw_data: pd.DataFrame, 
    norm_method: NormMethod = "pearson", 
    species: Species = "human",
    n_pca_components: int = N_PCA_COMPONENTS
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
    n_pca_components : int, default=N_PCA_COMPONENTS
        Number of principal components to retain
    
    Returns
    -------
    pd.DataFrame
        Processed data ready for clustering
    """
    # Always filter first
    clean_data = filter_data(raw_data, species=species)

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
    
    # Apply PCA
    print()
    print("Dimensionality Reduction (PCA)...")
    return apply_pca(normalized_data, n_components=n_pca_components)


def normalize_data_with_log_cpm(filtered_data: pd.DataFrame) -> pd.DataFrame:

    data = normalize_by_library_size(filtered_data)
    data = log_transform(data)

    return data


def filter_data(raw_data: pd.DataFrame, species: Species = "human") -> pd.DataFrame:
    """
    Runs the full filtering pipeline.
    """
    print(f"Filtering...")
    print(f"  Input: {raw_data.shape[0]} cells × {raw_data.shape[1]} genes")

    data = filter_low_magnitude_genes(raw_data)
    data = filter_high_apoptosis_cells(data, species=species)
    data = filter_high_rrna_cells(data, species=species)
    data = filter_high_mito_cells(data)

    print(f"  Output: {data.shape[0]} cells × {data.shape[1]} genes")
    return data
