import pandas as pd
from .filters import filter_rare_genes, filter_high_mito_cells
from .transforms import normalize_by_library_size, log_transform


def preprocess_sc_data(raw_data, min_cells=1, mito_percentile=95) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline.
    """
    print(f"--- Starting Preprocessing: Input Shape {raw_data.shape} ---")

    data = filter_rare_genes(raw_data, min_cells=min_cells)
    # First remove dead cells, because dead cells distort the library size normalization for the healthy cells.
    data = filter_high_mito_cells(data, percentile=mito_percentile)
    data = normalize_by_library_size(data)
    data = log_transform(data)

    print(f"--- Finished Preprocessing: Final Shape {data.shape} ---")
    return data
