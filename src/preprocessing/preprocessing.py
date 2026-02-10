import pandas as pd
from .types import Branch
from .filters import Species, filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform, normalize_data_with_pearson


def preprocess_data(raw_data: pd.DataFrame, branch: Branch = "pearson", species: Species = "human") -> pd.DataFrame:
    """
    Runs filtering, then only the requested normalization branch.
    """
    # Always filter first
    clean_data = filter_data(raw_data, species=species)

    # Selective normalization
    if branch == "log_cpm":
        print()
        print("Normalization (LogCPM)...")
        return normalize_data_with_log_cpm(clean_data)

    elif branch == "pearson":
        print()
        print("Normalization (Pearson Residuals)...")
        return normalize_data_with_pearson(clean_data)

    else:
        raise ValueError(f"Unknown preprocessing branch: {branch}")


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
