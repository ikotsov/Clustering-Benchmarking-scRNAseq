from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)

from src.types import InternalMetric


Matrix = np.ndarray | pd.DataFrame


def evaluate_clustering_internally(
    data: pd.DataFrame, labels_pred: pd.Series,
) -> dict[InternalMetric, float | None]:
    """
    Computes internal clustering metrics that do not require ground truth labels.
    Uses the raw feature data to assess cluster cohesion and separation.

    A metric is returned as None when it is undefined for the given labels,
    i.e. when there are fewer than 2 clusters, or every point forms its own
    singleton cluster.

    Metrics
    -------
    silhouette: float in [-1, 1], or None
        Higher is better. Measures how similar a cell is to its own cluster
        vs. other clusters.
    calinski_harabasz: float >= 0, or None
        Higher is better. Ratio of between-cluster to within-cluster dispersion.
    davies_bouldin: float >= 0, or None
        Lower is better. Average similarity between each cluster and its most
        similar cluster.
    """
    common_cells = data.index.intersection(labels_pred.index)
    X = data.loc[common_cells]
    y = labels_pred.loc[common_cells]

    return score_internal(X, y)


def compute_silhouette(X: Matrix, labels: ArrayLike, distances: np.ndarray | None = None) -> float:
    if distances is None:
        return float(silhouette_score(X, labels))
    return float(silhouette_score(distances, labels, metric="precomputed"))


def compute_calinski_harabasz(X: Matrix, labels: ArrayLike, distances: np.ndarray | None = None) -> float:
    return float(calinski_harabasz_score(X, labels))


def compute_davies_bouldin(X: Matrix, labels: ArrayLike, distances: np.ndarray | None = None) -> float:
    return float(davies_bouldin_score(X, labels))


INTERNAL_METRICS: dict[InternalMetric, Callable[[Matrix, ArrayLike, np.ndarray | None], float]] = {
    "silhouette": compute_silhouette,
    "calinski_harabasz": compute_calinski_harabasz,
    "davies_bouldin": compute_davies_bouldin,
}
LOWER_IS_BETTER: set[InternalMetric] = {"davies_bouldin"}
USES_DISTANCES: set[InternalMetric] = {"silhouette"}


def score_internal(
    X: Matrix,
    labels: ArrayLike,
    distances: np.ndarray | None = None,
    metrics: Iterable[InternalMetric] = INTERNAL_METRICS,
) -> dict[InternalMetric, float | None]:
    """
    Scores already aligned data and labels with the given internal metrics (all of INTERNAL_METRICS by default),
    all None when they are undefined for the labels (see are_internal_metrics_defined).

    distances : optional precomputed pairwise distances of X (see precompute_distances), so that scoring many
    labelings of the same data (e.g. the shuffles of the permutation test) computes them only once.
    """
    metrics = list(metrics)
    if not are_internal_metrics_defined(labels):
        return dict.fromkeys(metrics, None)
    return {metric: INTERNAL_METRICS[metric](X, labels, distances) for metric in metrics}


def precompute_distances(
    X: np.ndarray,
    metrics: Iterable[InternalMetric] = INTERNAL_METRICS,
) -> np.ndarray | None:
    """
    The pairwise distances of X to pass to score_internal when scoring many labelings of X with metrics,
    or None when none of the metrics uses them.
    """
    if USES_DISTANCES.isdisjoint(metrics):
        return None
    return pairwise_distances(X)


def are_internal_metrics_defined(labels: ArrayLike) -> bool:
    """
    Whether the internal metrics are defined for the labels: 2 <= n_clusters <= n_samples - 1.

    The bound is documented only for silhouette_score, but calinski_harabasz_score/davies_bouldin_score
    enforce it internally too (both call sklearn's check_number_of_labels), and all three raise
    ValueError outside it.
    """
    labels = np.asarray(labels)
    n_unique = len(np.unique(labels))
    return 2 <= n_unique < len(labels)
