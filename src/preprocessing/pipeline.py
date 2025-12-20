import pandas as pd
from .filters import filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform


def preprocess_sc_data(raw_data, mito_percentile=95) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline.
    """
    print(f"--- Starting Preprocessing: Input Shape {raw_data.shape} ---")

    data = filter_low_magnitude_genes(raw_data)
    data = filter_high_apoptosis_cells(data, percentile=mito_percentile)
    data = filter_high_rrna_cells(data, percentile=mito_percentile)
    data = filter_high_mito_cells(data, percentile=mito_percentile)

    data = normalize_by_library_size(data)
    data = log_transform(data)

    print(f"--- Finished Preprocessing: Final Shape {data.shape} ---")
    return data
