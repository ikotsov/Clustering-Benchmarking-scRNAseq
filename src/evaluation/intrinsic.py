import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering_internally(data: pd.DataFrame, labels_pred: pd.Series) -> dict[str, float]:
    """
    Computes internal clustering metrics that do not require ground truth labels.
    Uses the raw feature data to assess cluster cohesion and separation.

    Metrics
    -------
    silhouette: float in [-1, 1]
        Higher is better. Measures how similar a cell is to its own cluster
        vs. other clusters.
    calinski_harabasz: float >= 0
        Higher is better. Ratio of between-cluster to within-cluster dispersion.
    davies_bouldin: float >= 0
        Lower is better. Average similarity between each cluster and its most
        similar cluster.
    """
    common_cells = data.index.intersection(labels_pred.index)
    X = data.loc[common_cells]
    y = labels_pred.loc[common_cells]

    # These metrics require at least 2 distinct clusters
    n_unique = y.nunique()
    if n_unique < 2:
        return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": 0.0}

    return {
        "silhouette": float(silhouette_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
    }
