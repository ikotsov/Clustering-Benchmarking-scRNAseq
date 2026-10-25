"""
Permutation test of a clustering: how do its scores compare with those of random labels?

The predicted labels are shuffled across cells, which keeps the number and sizes of the clusters
but breaks any link to the data and to the ground truth. Scoring each shuffle with the same metrics
as the run itself gives a null distribution per metric, and an empirical p-value for the run.
"""
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from src.evaluation.extrinsic import EXTERNAL_METRICS, score_external
from src.evaluation.intrinsic import INTERNAL_METRICS, LOWER_IS_BETTER, are_internal_metrics_defined, score_internal
from src.types import PathLike


logger = logging.getLogger(__name__)

NULL_METRICS = [*EXTERNAL_METRICS, *INTERNAL_METRICS]


def compute_null_distribution(
    data: pd.DataFrame,
    labels_pred: pd.Series,
    labels_true: pd.Series,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """
    Scores n_permutations shuffles of the predicted labels with the external metrics (against labels_true)
    and the internal metrics (on data).

    Every cell of labels_pred must be in labels_true and in data (a KeyError otherwise);
    labels_true and data are reordered to match labels_pred.

    Returns
    -------
    null : pd.DataFrame
        One row per shuffle, one column per metric (NULL_METRICS). An internal metric that is undefined
        for the labels (fewer than 2 clusters, or only singletons) is NaN in every row, since shuffling
        keeps the number of clusters.
    """
    permuted = permute_labels(labels_pred.to_numpy(), n_permutations, seed)

    y_true = labels_true.loc[labels_pred.index].to_numpy()
    X = data.loc[labels_pred.index].to_numpy()

    # Silhouette needs every pairwise distance; computing them once saves recomputing them for each shuffle.
    distances = pairwise_distances(X) if are_internal_metrics_defined(labels_pred) else None

    rows = []
    for i, y_perm in enumerate(permuted):
        rows.append({**score_external(y_true, y_perm), **score_internal(X, y_perm, distances)})

        if (i + 1) % 100 == 0:
            logger.info("Scored %s/%s permutations", i + 1, n_permutations)

    # Undefined internal metrics (None) become NaN.
    return pd.DataFrame(rows, columns=pd.Index(NULL_METRICS)).astype(float)


def permute_labels(labels: np.ndarray, n_permutations: int, seed: int) -> np.ndarray:
    """Returns n_permutations independent shuffles of labels, one per row."""
    rng = np.random.default_rng(seed)
    return rng.permuted(np.tile(labels, (n_permutations, 1)), axis=1)


def summarize_null(observed: dict[str, float | None], null: pd.DataFrame) -> dict[str, float]:
    """
    Compares a run's observed metrics with their null distribution, as
    {metric}_null_mean and {metric}_pvalue for each metric of the null.

    The empirical p-value is (1 + #shuffles scoring at least as well as observed) / (1 + #shuffles),
    so it is never 0; its smallest value is 1 / (1 + #shuffles).
    A metric that is undefined for the run (None or NaN) gets NaN in both columns.
    """
    summary: dict[str, float] = {}
    for metric in null.columns:
        observed_value = observed.get(metric)
        values = null[metric].to_numpy()
        if observed_value is None or math.isnan(observed_value) or np.isnan(values).all():
            summary[f"{metric}_null_mean"] = math.nan
            summary[f"{metric}_pvalue"] = math.nan
            continue

        if metric in LOWER_IS_BETTER:
            n_as_good = int(np.sum(values <= observed_value))
        else:
            n_as_good = int(np.sum(values >= observed_value))

        summary[f"{metric}_null_mean"] = float(values.mean())
        summary[f"{metric}_pvalue"] = (1 + n_as_good) / (1 + len(values))
    return summary


def save_null_distribution(output_path: PathLike, null: pd.DataFrame, seed: int) -> None:
    """Saves a null distribution as a compressed .npz: values (shuffles x metrics), metrics (column names), seed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        values=null.to_numpy(dtype=np.float64),
        metrics=np.array(null.columns, dtype=str),
        seed=np.int64(seed),
    )
    logger.info("Saved null distribution (%s permutations) to: %s",
                len(null), output_path)


def load_null_distribution(path: PathLike) -> pd.DataFrame:
    """Loads a null distribution saved by save_null_distribution, one row per shuffle."""
    with np.load(path) as archive:
        return pd.DataFrame(archive["values"], columns=archive["metrics"].tolist())
