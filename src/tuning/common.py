import json
import os
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd

from src.evaluation import compute_ari, compute_nmi, compute_jaccard
from src.data_loading import load_csv_data
from src.clustering.registry import ClusteringAlgorithm
from src.types import NormMethod, PathLike


DEFAULT_NORM_METHOD: NormMethod = 'log_cpm'


def load_preprocessed_data(data_path: PathLike) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed file not found: {data_path}\n"
            "Run preprocessing first with run_preprocessing."
        )

    return load_csv_data(data_path)


ObjectiveMetric = Literal["ari", "nmi", "jaccard", "mean_external_score"]


ClusteringMetrics: TypeAlias = dict[ObjectiveMetric, float]


def compute_clustering_metrics(pred: pd.Series, true: pd.Series) -> ClusteringMetrics:
    metrics: ClusteringMetrics = {
        "ari": compute_ari(true, pred),
        "nmi": compute_nmi(true, pred),
        "jaccard": compute_jaccard(true, pred),
        "mean_external_score": 0.0,
    }
    metrics["mean_external_score"] = (
        metrics["ari"] + metrics["nmi"] + metrics["jaccard"]
    ) / 3.0
    return metrics


def save_tuning_results(
    output_path: PathLike,
    algorithm: ClusteringAlgorithm,
    norm_method: NormMethod,
    best_params: dict,
    best_value: float,
    objective_metric: ObjectiveMetric,
) -> str:
    """
    Save tuning results to a JSON file (clustering_params.json).
    Results are organized hierarchically by norm_method > algorithm.

    Args:
        output_path: Path to clustering_params.json (created if missing)
        algorithm: Clustering algorithm name
        norm_method: Normalization method used
        best_params: Best hyperparameters found
        best_value: Best objective metric value
        objective_metric: Which metric was optimized

    Returns:
        Path to the saved results file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing results or start fresh
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Ensure nested structure exists
    if norm_method not in all_results:
        all_results[norm_method] = {}

    # Add/update result for this algorithm
    all_results[norm_method][algorithm] = {
        "objective_metric": objective_metric,
        "best_params": best_params,
        "best_value": best_value,
    }

    # Save back to JSON
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return output_path
