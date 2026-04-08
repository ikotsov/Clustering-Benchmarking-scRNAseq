import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix


def evaluate_clustering_externally(labels_pred: pd.Series, labels_true: pd.Series) -> dict[str, float]:
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

    return {
        "ari": compute_ari(y_true, y_pred),
        "nmi": compute_nmi(y_true, y_pred),
        "jaccard": compute_jaccard(y_true, y_pred),
    }


def compute_ari(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(adjusted_rand_score(y_true, y_pred))


def compute_nmi(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred))


def compute_jaccard(y_true: pd.Series, y_pred: pd.Series) -> float:
    C = pair_confusion_matrix(y_true, y_pred)
    tp, fp, fn = int(C[1, 1]), int(C[0, 1]), int(C[1, 0])
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
