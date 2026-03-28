import json
import os
from typing import Literal, TypeAlias

import pandas as pd
from sklearn.metrics import accuracy_score

from src.evaluation import compute_ari, compute_nmi, compute_jaccard
from src.data_loading import load_csv_data
from src.clustering.registry import ClusteringAlgorithm
from src.types import NormMethod


DEFAULT_NORM_METHOD: NormMethod = 'log_cpm'


def load_preprocessed_data(
    accession: str,
    norm_method: NormMethod = DEFAULT_NORM_METHOD,
) -> pd.DataFrame:
    dataset_dir = os.path.join("data", accession)
    preprocessed_filename = f"{norm_method}_pca_preprocessed.csv.gz"
    preprocessed_file = os.path.join(
        dataset_dir, "results", preprocessed_filename)

    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(
            f"Preprocessed file not found: {preprocessed_filename}\n"
            f"Run preprocessing first with: run_preprocessing('{accession}', '{norm_method}')"
        )

    return load_csv_data(preprocessed_file)


ObjectiveMetric = Literal["ari", "nmi",
                          "jaccard", "accuracy", "mean_external_score"]


ClusteringMetrics: TypeAlias = dict[ObjectiveMetric, float]


def compute_clustering_metrics(pred: pd.Series, true: pd.Series) -> ClusteringMetrics:
    metrics: ClusteringMetrics = {
        "ari": compute_ari(true, pred),
        "nmi": compute_nmi(true, pred),
        "jaccard": compute_jaccard(true, pred),
        "accuracy": float(accuracy_score(true, pred)),
        "mean_external_score": 0.0,
    }
    metrics["mean_external_score"] = (
        metrics["ari"] + metrics["nmi"] + metrics["jaccard"]
    ) / 3.0
    return metrics


def save_tuning_results(
    accession: str,
    algorithm: ClusteringAlgorithm,
    norm_method: NormMethod,
    best_params: dict,
    best_value: float,
    objective_metric: ObjectiveMetric,
) -> str:
    """
    Save tuning results to a JSON file (tuning.json).
    Results are organized hierarchically by norm_method > algorithm.

    Args:
        accession: Dataset accession ID
        algorithm: Clustering algorithm name
        norm_method: Normalization method used
        best_params: Best hyperparameters found
        best_value: Best objective metric value
        objective_metric: Which metric was optimized

    Returns:
        Path to the saved results file
    """
    # Create output directory
    dataset_dir = os.path.join("data", accession)
    output_dir = os.path.join(dataset_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Path to tuning.json file
    output_path = os.path.join(output_dir, "tuning.json")

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
