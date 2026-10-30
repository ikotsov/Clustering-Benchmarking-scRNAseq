import json
import logging
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluation.biological.types import BiologicalComparisonRecord
from src.evaluation.permutation import load_null_distribution, summarize_null
from src.types import PathLike
from src.paths import get_params_label, get_pca_label


logger = logging.getLogger(__name__)


def save_run_result(
    algorithm: str,
    preprocessing: str,
    with_pca: bool,
    tuned: bool,
    metrics: dict[str, float | None],
    output_path: PathLike,
    n_cells: int,
    n_features: int,
    clustering_resources: dict[str, float],
) -> None:
    """
    Save the evaluation result of a single run into its own JSON file, overwriting any previous result.

    The file holds one run, and becomes one row of the results table (see aggregate_run_results):
        {
          "algorithm": "kmeans",
          "normalization": "pearson",
          "with_pca": true,
          "tuned": false,
          "n_cells": 1500,
          "n_features": 32,
          "metrics": {...},
          "clustering_resources": {"wall_time_s": ..., "cpu_time_s": ..., "rss_mb": ..., "rss_increase_mb": ...},
          "timestamp": "..."
        }

    Parameters
    ----------
    algorithm : str
        Clustering algorithm name
    preprocessing : str
        Normalization method used (e.g. "log_cpm", "pearson")
    with_pca : bool
        Whether PCA was applied
    tuned : bool
        Whether the algorithm ran with tuned parameters (False: its defaults)
    metrics : dict
        Dictionary of metric names and values (e.g., {"ari": 0.85, "nmi": 0.78})
    output_path : PathLike
        Path to the run's result file
    n_cells, n_features : int
        Shape of the clustered data (features: PCs with PCA, genes without)
    clustering_resources : dict
        Runtime and memory of the clustering call (see src.profiling.ResourceUsage)
    """
    run_entry = {
        "algorithm": algorithm,
        "normalization": preprocessing,
        "with_pca": with_pca,
        "tuned": tuned,
        "n_cells": n_cells,
        "n_features": n_features,
        "metrics": metrics,
        "clustering_resources": clustering_resources,
        "timestamp": datetime.now().isoformat(),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(run_entry, f, indent=2, allow_nan=False)

    logger.info(
        "Evaluation saved: %s -> %s -> %s -> %s in %s",
        algorithm,
        preprocessing,
        get_pca_label(with_pca),
        get_params_label(tuned),
        output_path,
    )


def aggregate_run_results(
    dataset: str,
    run_result_paths: Iterable[PathLike],
    results_path: PathLike,
    null_result_paths: Iterable[PathLike] | None = None,
) -> None:
    """
    Collect the per-run result files (see save_run_result) of a dataset into a results table (TSV)
    with one row per run:
        dataset  algorithm  normalization  pca  params  n_cells  n_features  ari  nmi  ...  clustering_wall_time_s  ...

    pca ("pca" / "no_pca") and params ("default" / "tuned") use the labels of the output paths
    (see src.paths), so the table can be filtered with the same values the files are named by.

    With null_result_paths (the null distribution of each run, see save_null_distribution, in the order
    of run_result_paths), each row also gets {metric}_null_mean and {metric}_pvalue (see summarize_null).

    The last columns are the runtime and memory of the clustering call.
    Runs saved before these were recorded get empty values, with a warning to rerun them.

    Rows keep the order of run_result_paths, and the table is rewritten from scratch,
    so it holds exactly the given runs.
    """
    run_result_paths = list(run_result_paths)
    null_paths = None if null_result_paths is None else list(null_result_paths)
    if null_paths is not None and len(null_paths) != len(run_result_paths):
        raise ValueError(
            f"Got {len(null_paths)} null results for {len(run_result_paths)} runs; expected one per run"
        )

    rows = []
    for i, run_result_path in enumerate(run_result_paths):
        with open(run_result_path, 'r') as f:
            run = json.load(f)
        null_summary = (
            summarize_null(run["metrics"], load_null_distribution(
                null_paths[i])) if null_paths is not None else {}
        )
        if "clustering_resources" not in run:
            logger.warning(
                "%s has no runtime or memory; rerun it to record them", run_result_path)
        rows.append({
            "dataset": dataset,
            **_run_columns(run),
            "n_cells": run.get("n_cells"),
            "n_features": run.get("n_features"),
            **run["metrics"],
            **null_summary,
            **{f"clustering_{name}": value for name, value in run.get("clustering_resources", {}).items()},
        })

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_path, sep="\t", index=False)

    logger.info("Aggregated %s runs into %s", len(rows), results_path)


def aggregate_run_labels(
    run_result_paths: Iterable[PathLike],
    run_labels_paths: Iterable[PathLike],
    labels_path: PathLike,
) -> None:
    """
    Collect the predicted labels of every run (run_labels_paths, in the order of run_result_paths)
    into one long table (TSV) with one row per cell and run:
        cell_id  algorithm  normalization  pca  params  cluster

    The run columns are those of the results table (see aggregate_run_results).
    """
    run_result_paths = list(run_result_paths)
    run_labels_paths = list(run_labels_paths)
    if len(run_labels_paths) != len(run_result_paths):
        raise ValueError(
            f"Got {len(run_labels_paths)} label files for {len(run_result_paths)} runs; expected one per run"
        )

    tables = []
    for run_result_path, run_labels_path in zip(run_result_paths, run_labels_paths):
        with open(run_result_path, 'r') as f:
            run = json.load(f)
        labels = pd.read_csv(run_labels_path, index_col=0).iloc[:, 0]
        tables.append(pd.DataFrame({
            "cell_id": labels.index,
            **_run_columns(run),
            "cluster": labels.to_numpy(),
        }))

    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(
        labels_path, sep="\t", index=False)

    logger.info("Aggregated the labels of %s runs into %s",
                len(tables), labels_path)


def _run_columns(run: dict) -> dict[str, str]:
    """The columns identifying a run (see save_run_result) in the results and labels tables."""
    return {
        "algorithm": run["algorithm"],
        "normalization": run["normalization"],
        "pca": get_pca_label(run["with_pca"]),
        "params": get_params_label(run["tuned"]),
    }


def save_biological_evaluation_results(
    dataset: str,
    algorithm: str,
    preprocessing: str,
    with_pca: bool,
    comparison_records: list[BiologicalComparisonRecord],
    output_dir: str,
) -> None:
    """Save biological evaluation results into biological_results.json."""
    results_file = os.path.join(output_dir, "biological_results.json")
    now_iso = datetime.now().isoformat()

    all_results: dict[str, object]
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            loaded = json.load(f)

        all_results = loaded
    else:
        all_results = {
            "dataset": dataset,
            "updated_at": now_iso,
            "runs": [],
        }

    runs = all_results["runs"]
    if not isinstance(runs, list):
        raise ValueError(
            "Invalid biological_results.json format: 'runs' must be a list."
        )

    run_entry = {
        "algorithm": algorithm,
        "normalization": preprocessing,
        "with_pca": with_pca,
        "comparison_records": comparison_records,
        "timestamp": now_iso,
    }

    replaced = False
    for idx, existing in enumerate(runs):
        if not isinstance(existing, dict):
            continue

        if (
            existing.get("algorithm") == algorithm
            and existing.get("normalization") == preprocessing
            and bool(existing.get("with_pca", False)) == with_pca
        ):
            runs[idx] = run_entry
            replaced = True
            break

    if not replaced:
        runs.append(run_entry)

    all_results.setdefault("dataset", dataset)
    all_results["updated_at"] = now_iso

    with open(results_file, "w") as f:
        json.dump(all_results, f, allow_nan=False)

    pca_key = get_pca_label(with_pca)
    logger.info(
        "Biological evaluation saved: %s -> %s -> %s in biological_results.json",
        algorithm,
        preprocessing,
        pca_key,
    )
