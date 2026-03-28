import os
from typing import Literal, TypeAlias

import pandas as pd
from sklearn.metrics import accuracy_score

from src.evaluation import compute_ari, compute_nmi, compute_jaccard
from src.data_loading import load_csv_data


NORM_METHOD = 'log_cpm'


def load_preprocessed_data(accession: str) -> pd.DataFrame:
    dataset_dir = os.path.join("data", accession)
    preprocessed_filename = f"{NORM_METHOD}_pca_preprocessed.csv.gz"
    preprocessed_file = os.path.join(
        dataset_dir, "results", preprocessed_filename)

    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(
            f"Preprocessed file not found: {preprocessed_filename}\n"
            f"Run preprocessing first with: run_preprocessing('{accession}')"
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
