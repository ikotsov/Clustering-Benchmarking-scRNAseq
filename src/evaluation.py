import pandas as pd
import json
import os
from datetime import datetime
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix


def evaluate_clustering(labels_pred: pd.Series, labels_true: pd.Series) -> dict[str, float]:
    """
    Compares predicted clusters against ground truth.
    Both Series must be aligned by the same index (Cell Barcodes).
    """
    # Ensure we only compare cells present in both (alignment)
    common_cells = labels_pred.index.intersection(labels_true.index)

    # Warn if many cells are missing
    if len(common_cells) < len(labels_pred) * 0.9:
        print(
            f"  ⚠ Warning: Only {len(common_cells)}/{len(labels_pred)} cells found in ground truth")

    y_pred = labels_pred.loc[common_cells]
    y_true = labels_true.loc[common_cells]

    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))

    C = pair_confusion_matrix(y_true, y_pred)
    tp, fp, fn = int(C[1, 1]), int(C[0, 1]), int(C[1, 0])
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"ari": ari, "nmi": nmi, "jaccard": jaccard}


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
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"), "davies_bouldin": float("nan")}

    return {
        "silhouette": float(silhouette_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
    }


def save_evaluation_results(
    dataset: str,
    algorithm: str,
    preprocessing: str,
    n_pca_components: int,
    metrics: dict[str, float],
    output_dir: str
) -> None:
    """
    Save evaluation results as JSON with metadata.

    Parameters
    ----------
    dataset : str
        Dataset accession ID
    algorithm : str
        Clustering algorithm name
    preprocessing : str
        Preprocessing branch used
    n_pca_components : int
        Number of PCA components used
    metrics : dict
        Dictionary of metric names and values (e.g., {"ari": 0.85, "nmi": 0.78})
    output_dir : str
        Directory to save the results
    """
    results = {
        "dataset": dataset,
        "algorithm": algorithm,
        "preprocessing": preprocessing,
        "n_pca_components": n_pca_components,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    filename = f"{preprocessing}_pca{n_pca_components}_{algorithm}_evaluation.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Evaluation saved: {filename}")
