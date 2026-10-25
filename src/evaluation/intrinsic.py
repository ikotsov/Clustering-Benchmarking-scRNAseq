import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


INTERNAL_METRICS = ["silhouette", "calinski_harabasz", "davies_bouldin"]
LOWER_IS_BETTER = {"davies_bouldin"}


def evaluate_clustering_internally(data: pd.DataFrame, labels_pred: pd.Series) -> dict[str, float | None]:
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


def score_internal(
    X: np.ndarray | pd.DataFrame,
    labels: ArrayLike,
    distances: np.ndarray | None = None,
) -> dict[str, float | None]:
    """
    Scores already aligned data and labels with every internal metric (INTERNAL_METRICS),
    all None when they are undefined for the labels (see are_internal_metrics_defined).

    distances : optional precomputed pairwise distances of X, so that scoring many labelings of
    the same data (e.g. the shuffles of the permutation test) computes them only once.
    """
    if not are_internal_metrics_defined(labels):
        return dict.fromkeys(INTERNAL_METRICS, None)

    silhouette = (
        silhouette_score(X, labels) if distances is None
        else silhouette_score(distances, labels, metric="precomputed")
    )
    return {
        "silhouette": float(silhouette),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


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
