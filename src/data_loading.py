import json
import os
import pandas as pd
from typing import cast

from src.dataset_config import DatasetConfig
from src.types import PathLike
from src.preprocessing.types import PreprocessingConfig


def load_csv_data(file_path: PathLike) -> pd.DataFrame:
    return pd.read_csv(file_path, compression='gzip', index_col=0)


def parse_preprocessing_config(config: DatasetConfig) -> PreprocessingConfig:
    defaults = PreprocessingConfig()

    raw_preprocessing = config.get("preprocessing")
    if not isinstance(raw_preprocessing, dict):
        return defaults

    cutoffs = cast(dict[str, object], raw_preprocessing)

    mito_value = cutoffs.get("mito_threshold")
    min_gene_max_count_value = cutoffs.get("min_gene_max_count")

    mito_threshold = float(mito_value) if isinstance(
        mito_value, (int, float)) else defaults.mito_threshold
    min_gene_max_count = int(min_gene_max_count_value) if isinstance(
        min_gene_max_count_value, (int, float)) else defaults.min_gene_max_count

    count_value = cutoffs.get("count_threshold")
    gene_value = cutoffs.get("gene_threshold")

    count_threshold = float(count_value) if isinstance(
        count_value, (int, float)) else defaults.count_threshold
    gene_threshold = int(gene_value) if isinstance(
        gene_value, (int, float)) else defaults.gene_threshold

    return PreprocessingConfig(
        mito_threshold=mito_threshold,
        min_gene_max_count=min_gene_max_count,
        count_threshold=count_threshold,
        gene_threshold=gene_threshold,
    )


ClusteringParamValue = int | float | str | bool | None


def parse_clustering_params(config: DatasetConfig, algorithm: str) -> dict[str, ClusteringParamValue]:
    raw_clustering = config.get("clustering_params")
    if not isinstance(raw_clustering, dict):
        return {}

    raw_algo_params = raw_clustering.get(algorithm)
    if not isinstance(raw_algo_params, dict):
        return {}

    parsed: dict[str, ClusteringParamValue] = {}
    for key, value in raw_algo_params.items():
        if not isinstance(key, str):
            continue

        if isinstance(value, (int, float, str, bool)) or value is None:
            parsed[key] = value

    return parsed


def load_tuned_params(params_path: PathLike | None) -> dict[str, ClusteringParamValue]:
    """
    Loads the best parameters from a tuning result file (see save_tuning_results).
    Returns no parameters when params_path is None, i.e. for algorithms that are not tuned.
    """
    if params_path is None:
        return {}

    with open(params_path, "r") as f:
        loaded = json.load(f)

    raw_best_params = loaded.get("best_params", {})
    if not isinstance(raw_best_params, dict):
        return {}

    parsed: dict[str, ClusteringParamValue] = {}
    for key, value in raw_best_params.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            parsed[key] = value

    return parsed


def load_ground_truth_labels(labels_path: PathLike) -> pd.Series:
    """
    Load ground truth labels from saved CSV file.

    Parameters
    ----------
    labels_path : PathLike
        Path to the ground truth labels CSV (cell identifiers in the first column)

    Returns
    -------
    labels : pd.Series
        Series with cell identifiers as index and true labels as values
    """
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
