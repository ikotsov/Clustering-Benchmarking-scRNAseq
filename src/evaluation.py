import pandas as pd
import json
import os
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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

    return {"ari": ari, "nmi": nmi}


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
