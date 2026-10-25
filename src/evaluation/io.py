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
) -> None:
    """
    Save the evaluation result of a single run into its own JSON file, overwriting any previous result.

    The file holds one run, and becomes one row of the results table (see aggregate_run_results):
        {
          "algorithm": "kmeans",
          "normalization": "pearson",
          "with_pca": true,
          "tuned": false,
          "metrics": {...},
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
    """
    run_entry = {
        "algorithm": algorithm,
        "normalization": preprocessing,
        "with_pca": with_pca,
        "tuned": tuned,
        "metrics": metrics,
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
        dataset  algorithm  normalization  with_pca  tuned  ari  nmi  ...  timestamp

    With null_result_paths (the null distribution of each run, see save_null_distribution, in the order
    of run_result_paths), each row also gets {metric}_null_mean and {metric}_pvalue (see summarize_null).

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
            summarize_null(run["metrics"], load_null_distribution(null_paths[i])) if null_paths is not None else {}
        )
        rows.append({
            "dataset": dataset,
            "algorithm": run["algorithm"],
            "normalization": run["normalization"],
            "with_pca": run["with_pca"],
            "tuned": run["tuned"],
            **run["metrics"],
            **null_summary,
            "timestamp": run["timestamp"],
        })

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_path, sep="\t", index=False)

    logger.info("Aggregated %s runs into %s", len(rows), results_path)


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
