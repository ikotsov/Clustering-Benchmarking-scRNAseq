"""
Writes a dataset in the standard form the workflow starts from.

Every dataset's prepare script (scripts/prepare/<accession>.py) turns its own source files into:
- a counts matrix: cells x genes, raw counts, cell IDs as the index (gzipped CSV)
- ground truth labels: one label per cell of the counts matrix (CSV: cell_id, true_label)
"""
import logging
from pathlib import Path

import pandas as pd

from src.types import PathLike


logger = logging.getLogger(__name__)


def save_prepared_dataset(
    counts: pd.DataFrame,
    labels: pd.Series,
    counts_path: PathLike,
    labels_path: PathLike,
) -> None:
    """
    Checks that counts (cells x genes) and labels describe the same cells, then saves both.
    Labels are saved in the row order of the counts matrix.
    """
    _check_unique(counts.index, "cell IDs in the counts matrix")
    _check_unique(counts.columns, "genes in the counts matrix")
    _check_unique(labels.index, "cell IDs in the labels")

    unlabeled = counts.index.difference(labels.index)
    not_in_counts = labels.index.difference(counts.index)
    if len(unlabeled) or len(not_in_counts):
        raise ValueError(
            f"Counts and labels describe different cells: {len(unlabeled)} cells have no label "
            f"(e.g. {list(unlabeled[:3])}), {len(not_in_counts)} labeled cells are not in the counts "
            f"(e.g. {list(not_in_counts[:3])})"
        )

    counts = counts.rename_axis("cell_id")
    labels = labels.loc[counts.index].rename("true_label")

    Path(counts_path).parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(counts_path, compression="gzip")
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path, header=True)

    logger.info("Saved counts: %s (%s cells x %s genes)", counts_path, counts.shape[0], counts.shape[1])
    logger.info("Saved labels: %s (%s labels: %s)", labels_path, labels.nunique(), labels.value_counts().to_dict())


def _check_unique(index: pd.Index, what: str) -> None:
    duplicated = index[index.duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"Duplicate {what}: {len(duplicated)} (e.g. {list(duplicated[:3])})")
