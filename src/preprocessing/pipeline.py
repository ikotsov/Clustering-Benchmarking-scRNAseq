import os
import pandas as pd
from ..data_loading import load_10x_data
from .filters import filter_high_mito_cells, filter_high_rrna_cells, filter_high_apoptosis_cells, filter_low_magnitude_genes
from .transforms import normalize_by_library_size, log_transform, normalize_data_with_pearson


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    data_path = os.path.join(project_root, "data", "10x")
    output_dir = os.path.join(project_root, "data")

    raw_data = load_10x_data(data_path)
    preprocessed_data = preprocess_data(raw_data)

    os.makedirs(output_dir, exist_ok=True)

    print("Saving pearson_data.csv...")
    # index=True is usually good for 10x data to keep cell barcodes (if they are the index)
    preprocessed_data["pearson"].to_csv(os.path.join(
        output_dir, "pearson_data.csv"), index=True)

    print("Saving log_cpm_data.csv...")
    preprocessed_data["log_cpm"].to_csv(os.path.join(
        output_dir, "log_cpm_data.csv"), index=True)

    print(f"Done! Files saved.")


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
    pearson_data = normalize_data_with_pearson(clean_data)

    return {
        "log_cpm": log_cpm_data,
        "pearson": pearson_data
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


if __name__ == "__main__":
    main()
