import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd

from src.evaluation import compute_ari, compute_nmi, compute_jaccard
from src.data_loading import load_csv_data
from src.clustering.registry import ClusteringAlgorithm
from src.paths import get_pca_label
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
    with_pca: bool,
    best_params: dict,
    best_value: float,
    objective_metric: ObjectiveMetric,
) -> PathLike:
    """
    Save the tuning result of one algorithm + normalization + PCA option into its own JSON file,
    overwriting any previous result.

    Args:
        output_path: Path to the tuning result file
        algorithm: Clustering algorithm name
        norm_method: Normalization method used
        with_pca: Whether the PCA representation was tuned on
        best_params: Best hyperparameters found
        best_value: Best objective metric value
        objective_metric: Which metric was optimized

    Returns:
        Path to the saved results file
    """
    result = {
        "algorithm": algorithm,
        "normalization": norm_method,
        "with_pca": with_pca,
        "objective_metric": objective_metric,
        "best_params": best_params,
        "best_value": best_value,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return output_path


def aggregate_tuning_results(
    tuning_result_paths: Iterable[PathLike],
    output_path: PathLike,
) -> None:
    """
    Collect per-job tuning result files (see save_tuning_results) into clustering_params.json,
    organized hierarchically by norm_method > pca > algorithm. The file is rewritten from scratch,
    so it holds exactly the given results.
    """
    all_results: dict[str, dict[str, dict[str, dict]]] = {}
    for tuning_result_path in tuning_result_paths:
        with open(tuning_result_path, "r") as f:
            result = json.load(f)

        pca_label = get_pca_label(result["with_pca"])
        all_results.setdefault(result["normalization"], {}).setdefault(pca_label, {})[result["algorithm"]] = {
            "objective_metric": result["objective_metric"],
            "best_params": result["best_params"],
            "best_value": result["best_value"],
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
