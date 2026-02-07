import pandas as pd
from .filters import filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform, normalize_data_with_pearson


def preprocess_data(raw_data: pd.DataFrame, branch: str = "pearson") -> pd.DataFrame:
    """
    Runs filtering, then only the requested normalization branch.
    """
    # Always filter first
    clean_data = filter_data(raw_data)

    # Selective normalization
    if branch == "log_cpm":
        print("--- Running LogCPM normalization ---")
        return normalize_data_with_log_cpm(clean_data)

    elif branch == "pearson":
        print("--- Running NB Pearson Residuals ---")
        return normalize_data_with_pearson(clean_data)

    else:
        raise ValueError(f"Unknown preprocessing branch: {branch}")


def normalize_data_with_log_cpm(filtered_data: pd.DataFrame) -> pd.DataFrame:

    data = normalize_by_library_size(filtered_data)
    data = log_transform(data)

    return data


def filter_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full filtering pipeline.
    """
    print(f"--- Starting Filtering: Input Shape {raw_data.shape} ---")

    data = filter_low_magnitude_genes(raw_data)
    data = filter_high_apoptosis_cells(data)
    data = filter_high_rrna_cells(data)
    data = filter_high_mito_cells(data)

    print(f"--- Finished Filtering: Final Shape {data.shape} ---")
    return data
