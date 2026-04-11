import pandas as pd
import scanpy as sc
from .types import PreprocessingConfig
from .filters import filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform, normalize_data_with_pearson
from src.types import NormMethod, Species


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
        Raw gene expression data (cells × genes)
    norm_method : NormMethod, default="pearson"
        Normalization method to use ("log_cpm" or "pearson")
    species : Species, default="human"
        Species for filtering (affects mitochondrial, ribosomal, apoptosis genes)
    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Processed data (HVG space) ready for optional downstream PCA and selected HVG names
    """
    # Always filter first
    clean_data = filter_data(
        raw_data, config=preprocessing_config, species=species)

    # Selective normalization
    if norm_method == "log_cpm":
        print()
        print("Normalization (LogCPM)...")
        normalized_data, hvg_genes = normalize_data_with_log_cpm(clean_data)

    elif norm_method == "pearson":
        print()
        print("Normalization (Pearson Residuals)...")
        normalized_data, hvg_genes = normalize_data_with_pearson(clean_data)

    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    return normalized_data, hvg_genes


# In Seurat, 2,000 HVGs is default.
N_HVG = 2_000


def normalize_data_with_log_cpm(filtered_data: pd.DataFrame, n_hvg: int = N_HVG) -> tuple[pd.DataFrame, list[str]]:
    print(f"  • Selecting top {n_hvg} variable genes")
    adata = sc.AnnData(filtered_data)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3_paper",
        n_top_genes=n_hvg,
    )
    hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()

    data = filtered_data.loc[:, hvg_genes]

    data = normalize_by_library_size(data)
    data = log_transform(data)

    return data, hvg_genes


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
