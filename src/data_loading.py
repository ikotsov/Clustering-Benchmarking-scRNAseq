import yaml
import os
import pandas as pd
from typing import cast, NotRequired, TypedDict

from src.types import Species
from src.preprocessing.types import PreprocessingConfig


def load_csv_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, compression='gzip', index_col=0)


class DatasetConfig(TypedDict):
    species: Species
    n_clusters: int
    preprocessing: NotRequired[dict[str, float | int]]


def load_dataset_config(dataset_dir: str) -> DatasetConfig:
    default_config: DatasetConfig = {
        "species": "human",
        "n_clusters": 0,
    }

    config_path = os.path.join(dataset_dir, "config.yaml")
    if not os.path.exists(config_path):
        return default_config

    with open(config_path, "r") as f:
        loaded = yaml.safe_load(f)

    if not isinstance(loaded, dict):
        return default_config

    return cast(DatasetConfig, loaded)


def parse_preprocessing_config(config: DatasetConfig) -> PreprocessingConfig:
    defaults = PreprocessingConfig()

    raw_preprocessing = config.get("preprocessing")
    if not isinstance(raw_preprocessing, dict):
        return defaults

    cutoffs = cast(dict[str, object], raw_preprocessing)

    mito_value = cutoffs.get("mito_threshold")
    rrna_value = cutoffs.get("rrna_threshold")
    apoptosis_value = cutoffs.get("apoptosis_threshold")
    gene_magnitude_value = cutoffs.get("gene_magnitude_threshold")

    mito_threshold = float(mito_value) if isinstance(
        mito_value, (int, float)) else defaults.mito_threshold
    rrna_threshold = float(rrna_value) if isinstance(
        rrna_value, (int, float)) else defaults.rrna_threshold
    apoptosis_threshold = float(apoptosis_value) if isinstance(
        apoptosis_value, (int, float)) else defaults.apoptosis_threshold
    gene_magnitude_threshold = int(gene_magnitude_value) if isinstance(
        gene_magnitude_value, (int, float)) else defaults.gene_magnitude_threshold

    return PreprocessingConfig(
        mito_threshold=mito_threshold,
        rrna_threshold=rrna_threshold,
        apoptosis_threshold=apoptosis_threshold,
        gene_magnitude_threshold=gene_magnitude_threshold,
    )


def load_ground_truth_labels(dataset_dir: str) -> pd.Series:
    """
    Load ground truth labels from saved CSV file.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory

    Returns
    -------
    labels : pd.Series
        Series with cell identifiers as index and true labels as values
    """
    labels_path = os.path.join(dataset_dir, "raw", "ground_truth_labels.csv")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Ground truth labels not found: {labels_path}\n"
            "Please extract labels from metadata first using the exploration notebook."
        )

    labels = pd.read_csv(labels_path, index_col=0)

    # Reads both DataFrames and Series, but we only want the first column if it's a DataFrame
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]  # all rows, first column

    labels.name = "true_label"
    return labels
