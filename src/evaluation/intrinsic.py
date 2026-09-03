import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


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

    # These metrics are only defined for 2 <= n_clusters <= n_samples - 1
    n_unique = y.nunique()
    if n_unique < 2 or n_unique >= len(y):
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    return {
        "silhouette": float(silhouette_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
    }
