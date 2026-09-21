import math
from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score

from .types import (
    BiologicalComparisonRecord,
    BiologicalMetricsRecord,
    EnrichmentSetName,
    FoldEnrichmentMetrics,
    MembershipMetrics,
    RandomSummaryStats,
    TailStats,
)

# Metrics where a *higher* value is more biologically meaningful, so the
# significance test asks "is the observed value higher than random chance?".
RIGHT_TAIL_METRICS: tuple[str, ...] = (
    "jaccard", "precision", "recall", "specificity", "fold_enrichment",
)
# Metrics where a *lower* value is more meaningful (a false-positive rate
# should be below what random gene selection produces), so the test asks
# "is the observed value lower than random chance?".
LEFT_TAIL_METRICS: tuple[str, ...] = ("fpr",)
# Raw counts that only get a descriptive summary, not a z-score/p-value/CI significance test.
RANDOM_SUMMARY_ONLY_METRICS: tuple[str, ...] = (
    "tp", "fp", "fn", "tn", "intersection", "expected",
)
ALL_NULL_TRACKED_METRICS: tuple[str, ...] = (
    RIGHT_TAIL_METRICS + LEFT_TAIL_METRICS + RANDOM_SUMMARY_ONLY_METRICS
)


def compute_run_metrics(
    cluster_marker_genes: list[str],
    enrichment_sets: dict[EnrichmentSetName, dict[str, set[str]]],
    all_genes: set[str],
    cluster_id: str,
    cluster_size: int,
    sample_type: str,
    sample_index: int,
    sample_seed: int,
    run_hvg_count: int,
) -> list[BiologicalMetricsRecord]:
    """Compute raw biological metrics for one cluster of one run. """
    selected_cluster_markers = set(cluster_marker_genes)

    universe = sorted(all_genes)
    selected_cluster_marker_vector = _binary_membership_vector(
        universe,
        selected_cluster_markers,
    )

    results: list[BiologicalMetricsRecord] = []

    for enrichment_set_name, marker_sets_by_cell_type in enrichment_sets.items():
        for cell_type, reference_marker_genes in marker_sets_by_cell_type.items():
            marker_set = set(reference_marker_genes)
            y_true = _binary_membership_vector(universe, marker_set)

            run_metrics = _compute_membership_metrics(
                y_true=y_true,
                y_pred=selected_cluster_marker_vector,
            )
            run_fold = _compute_fold_enrichment(
                selected_genes=selected_cluster_markers,
                marker_genes=marker_set,
                all_genes=all_genes,
            )

            results.append(
                {
                    "enrichment_set": enrichment_set_name,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "marker_genes": cluster_marker_genes,
                    "n_marker_genes": len(cluster_marker_genes),
                    "cell_type": cell_type,
                    "sample_type": sample_type,
                    "sample_index": sample_index,
                    "sample_seed": sample_seed,
                    "run_hvg_count": run_hvg_count,
                    "jaccard": run_metrics["jaccard"],
                    "tp": run_metrics["tp"],
                    "fp": run_metrics["fp"],
                    "fn": run_metrics["fn"],
                    "tn": run_metrics["tn"],
                    "precision": run_metrics["precision"],
                    "recall": run_metrics["recall"],
                    "specificity": run_metrics["specificity"],
                    "fpr": run_metrics["fpr"],
                    "intersection": run_fold["intersection"],
                    "expected": run_fold["expected"],
                    "fold_enrichment": run_fold["fold_enrichment"],
                }
            )

    return results


def attach_tail_statistics(
    observed_records: list[BiologicalMetricsRecord],
    sampled_records: list[BiologicalMetricsRecord],
) -> list[BiologicalComparisonRecord]:
    """Test each observed metric against the null distribution of sampled runs.

    `sampled_records` are the same metrics computed on many KDE-matched
    random gene sets instead of the real HVGs, they are the "what would this
    look like by chance" baseline that each observed value is tested against.

    A random/sampled rerun can produce a different number of clusters than
    the observed run (algorithms like leiden/hdbscan/optics choose their own
    cluster count, and even a fixed cluster count doesn't give clusters a
    stable identity across reruns). So "observed cluster 2" cannot be matched
    to one specific cluster in a given random rerun.

    Instead, for every random rerun we take the *best* (max, for metrics
    where higher is more meaningful) value across that rerun's own clusters,
    per reference cell type. This answers: "what is the strongest alignment
    to this cell type that a random gene subset could produce by chance, in
    one rerun?". Doing this for all `n_samples` reruns gives exactly one null
    value per rerun per cell type. For metrics where *lower* is more meaningful 
    (false-positive rate), we take the rerun's *minimum* instead, for the same 
    reason in the opposite direction.
    """
    null_pools = _build_null_pools(sampled_records)

    enriched_records: list[BiologicalComparisonRecord] = []
    for record in observed_records:
        pool_key = (record["enrichment_set"], record["cell_type"])
        metric_pools = null_pools.get(pool_key, {})

        enriched: BiologicalComparisonRecord = dict(record)
        for metric_name in RIGHT_TAIL_METRICS:
            random_values = metric_pools.get(metric_name, [])
            observed_value = _as_optional_float(record.get(metric_name))
            stats = _right_tail_stats(observed_value, random_values)
            enriched.update(_prefix_stats(stats, metric_name))
            enriched.update(_random_summary(metric_name, random_values))
        for metric_name in LEFT_TAIL_METRICS:
            random_values = metric_pools.get(metric_name, [])
            observed_value = _as_optional_float(record.get(metric_name))
            stats = _left_tail_stats(observed_value, random_values)
            enriched.update(_prefix_stats(stats, metric_name))
            enriched.update(_random_summary(metric_name, random_values))
        for metric_name in RANDOM_SUMMARY_ONLY_METRICS:
            random_values = metric_pools.get(metric_name, [])
            enriched.update(_random_summary(metric_name, random_values))

        enriched_records.append(enriched)

    return enriched_records


def _as_optional_float(value: object) -> float | None:
    """Narrow a dynamically-looked-up record field to the numeric type it
    always actually holds (`.get()` with a non-literal key can't be typed
    more precisely than `object` on a TypedDict)."""
    return None if value is None else cast(float, value)


def _build_null_pools(
    sampled_records: list[BiologicalMetricsRecord],
) -> dict[tuple[EnrichmentSetName, str], dict[str, list[float | None]]]:
    """Reduce every sampled run's clusters to one null value per metric.

    Groups sampled-run records by `(enrichment_set, cell_type)`, then within
    each group by `sample_index` (one random rerun can contribute several
    cluster records), and reduces each rerun's cluster records to a single
    representative value per metric, max for `RIGHT_TAIL_METRICS`/
    `RANDOM_SUMMARY_ONLY_METRICS`, min for `LEFT_TAIL_METRICS`. See
    `attach_tail_statistics` for why this pooling is needed.
    """
    records_by_key_and_sample: dict[
        tuple[EnrichmentSetName, str], dict[int, dict[str, list[float | None]]]
    ] = {}
    for record in sampled_records:
        key = (record["enrichment_set"], record["cell_type"])
        sample_index = record["sample_index"]
        by_sample = records_by_key_and_sample.setdefault(key, {})
        metric_values = by_sample.setdefault(
            sample_index, {name: [] for name in ALL_NULL_TRACKED_METRICS}
        )
        for metric_name in ALL_NULL_TRACKED_METRICS:
            metric_values[metric_name].append(_as_optional_float(record.get(metric_name)))

    null_pools: dict[tuple[EnrichmentSetName, str], dict[str, list[float | None]]] = {}
    for key, by_sample in records_by_key_and_sample.items():
        pooled: dict[str, list[float | None]] = {
            name: [] for name in ALL_NULL_TRACKED_METRICS
        }
        for metric_values in by_sample.values():
            for metric_name in ALL_NULL_TRACKED_METRICS:
                clean_values = [
                    value
                    for value in metric_values[metric_name]
                    if value is not None and not math.isnan(value)
                ]
                if not clean_values:
                    pooled[metric_name].append(None)
                elif metric_name in LEFT_TAIL_METRICS:
                    pooled[metric_name].append(min(clean_values))
                else:
                    pooled[metric_name].append(max(clean_values))
        null_pools[key] = pooled

    return null_pools


def _compute_membership_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MembershipMetrics:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    fpr = float(1.0 - specificity) if pd.notna(specificity) else np.nan

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "jaccard": None if pd.isna(jaccard_score(y_true, y_pred, zero_division=0)) else float(jaccard_score(y_true, y_pred, zero_division=0)),
        "precision": None if pd.isna(precision_score(y_true, y_pred, zero_division=0)) else float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": None if pd.isna(recall_score(y_true, y_pred, zero_division=0)) else float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": None if pd.isna(specificity) else float(specificity),
        "fpr": None if pd.isna(fpr) else float(fpr),
    }


def _compute_fold_enrichment(
    selected_genes: set[str],
    marker_genes: set[str],
    all_genes: set[str],
) -> FoldEnrichmentMetrics:
    intersection = len(selected_genes & marker_genes)
    universe_size = len(all_genes)
    expected = (
        (len(selected_genes) * len(marker_genes)) / universe_size
        if universe_size > 0
        else np.nan
    )
    fold_enrichment = (
        intersection /
        expected if pd.notna(expected) and expected > 0 else np.nan
    )

    return {
        "intersection": int(intersection),
        "expected": None if pd.isna(expected) else float(expected),
        "fold_enrichment": None if pd.isna(fold_enrichment) else float(fold_enrichment),
    }


def _binary_membership_vector(universe_genes: list[str], selected_genes: set[str]) -> np.ndarray:
    return np.asarray([1 if gene in selected_genes else 0 for gene in universe_genes], dtype=int)


def _prefix_stats(stats: TailStats, metric_name: str) -> dict[str, float | None]:
    return {
        f"{metric_name}_z_score": stats["z_score"],
        f"{metric_name}_p_value": stats["p_value"],
        f"{metric_name}_ci_lower": stats["ci_lower"],
        f"{metric_name}_ci_upper": stats["ci_upper"],
    }


def _compute_random_summary(values: Sequence[int | float | None]) -> RandomSummaryStats:
    clean_values = [
        float(value) for value in values if value is not None and not math.isnan(value)
    ]
    if len(clean_values) == 0:
        return {"mean": None, "std": None, "n": 0}

    values_array = np.asarray(clean_values)
    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "n": int(len(clean_values)),
    }


def _random_summary(
    metric_name: str,
    values: Sequence[int | float | None],
) -> dict[str, float | int | None]:
    summary = _compute_random_summary(values)
    return {
        f"random_{metric_name}_mean": summary["mean"],
        f"random_{metric_name}_std": summary["std"],
        f"random_{metric_name}_n": summary["n"],
    }


def _right_tail_stats(
    observed: float | None,
    random_values: Sequence[float | None],
) -> TailStats:
    return _compute_tail_stats(observed, random_values, direction="right")


def _left_tail_stats(observed: float | None, random_values: list[float | None]) -> TailStats:
    return _compute_tail_stats(observed, random_values, direction="left")


def _compute_tail_stats(
    observed: float | None,
    random_values: Sequence[float | None],
    direction: Literal["right", "left"],
) -> TailStats:
    clean_random = [
        float(value) for value in random_values if value is not None and not math.isnan(value)
    ]
    if observed is None or math.isnan(observed) or len(clean_random) == 0:
        return {"z_score": None, "p_value": None, "ci_lower": None, "ci_upper": None}

    random_array = np.asarray(clean_random)
    z_score = (
        (float(observed) - float(np.mean(random_array))) /
        float(np.std(random_array))
        if float(np.std(random_array)) != 0
        else np.nan
    )
    if direction == "right":
        p_value = (float(np.sum(random_array >= float(observed))) +
                   1.0) / (len(random_array) + 1.0)
    else:
        p_value = (float(np.sum(random_array <= float(observed))) +
                   1.0) / (len(random_array) + 1.0)
    ci_lower, ci_upper = np.percentile(random_array, [2.5, 97.5])
    return {
        "z_score": None if pd.isna(z_score) else float(z_score),
        "p_value": None if pd.isna(p_value) else float(p_value),
        "ci_lower": None if pd.isna(ci_lower) else float(ci_lower),
        "ci_upper": None if pd.isna(ci_upper) else float(ci_upper),
    }
