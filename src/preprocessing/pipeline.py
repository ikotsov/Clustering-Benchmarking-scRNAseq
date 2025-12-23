import pandas as pd
from .filters import filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform


def preprocess_data(raw_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Runs the full preprocessing pipeline, including filtering and normalization.
    """
    clean_data = filter_data(raw_data)

    # BRANCH A: LogCPM
    print("--- Branch A: Running LogCPM ---")
    log_cpm_data = normalize_data_with_log_cpm(clean_data)

    # BRANCH B: NB Fitting (Pearson)
    print("--- Branch B: Running NB Pearson Residuals ---")
    # pearson_data = normalize_data_with_pearson(clean_data)

    return {
        "log_cpm": log_cpm_data,
        # "pearson": pearson_data
    }


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
