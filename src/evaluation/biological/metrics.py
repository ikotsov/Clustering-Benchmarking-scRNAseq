import math
from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd

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
    """Compute raw biological metrics for one gene set (a cluster's markers, or a random set) vs every reference set."""
    selected_cluster_markers = set(cluster_marker_genes)

    results: list[BiologicalMetricsRecord] = []

    for enrichment_set_name, marker_sets_by_cell_type in enrichment_sets.items():
        for cell_type, reference_marker_genes in marker_sets_by_cell_type.items():
            marker_set = set(reference_marker_genes)

            run_metrics = _compute_membership_metrics(
                reference_genes=marker_set,
                selected_genes=selected_cluster_markers,
                all_genes=all_genes,
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
    """Test each observed metric against the null distribution of its own cluster.

    `sampled_records` hold the same metrics for random gene sets drawn for one cluster
    (same size as its marker set, expression-matched to it), tagged with that cluster's id.
    Each observed record (enrichment set, cluster, cell type) is tested against the
    `n_samples` values of the same (enrichment set, cluster, cell type): right tail for
    metrics where higher is more meaningful, left tail for FPR.
    """
    null_pools = _build_null_pools(sampled_records)

    enriched_records: list[BiologicalComparisonRecord] = []
    for record in observed_records:
        pool_key = (record["enrichment_set"], record["cluster_id"], record["cell_type"])
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


NullPoolKey = tuple[EnrichmentSetName, str, str]


def _build_null_pools(
    sampled_records: list[BiologicalMetricsRecord],
) -> dict[NullPoolKey, dict[str, list[float | None]]]:
    """Collect the random values of every metric per (enrichment set, cluster, cell type).

    One value per random set: each random set is scored once against each reference set.
    """
    null_pools: dict[NullPoolKey, dict[str, list[float | None]]] = {}
    for record in sampled_records:
        key = (record["enrichment_set"], record["cluster_id"], record["cell_type"])
        pooled = null_pools.setdefault(key, {name: [] for name in ALL_NULL_TRACKED_METRICS})
        for metric_name in ALL_NULL_TRACKED_METRICS:
            pooled[metric_name].append(_as_optional_float(record.get(metric_name)))

    return null_pools


def _compute_membership_metrics(
    reference_genes: set[str],
    selected_genes: set[str],
    all_genes: set[str],
) -> MembershipMetrics:
    """Confusion counts of `selected_genes` (prediction) vs `reference_genes` (truth) within `all_genes`.

    Ratios with an empty denominator are 0 (as sklearn's zero_division=0), except
    specificity/FPR, which are None when the universe has no negatives.
    """
    tp = len(selected_genes & reference_genes)
    fp = len(selected_genes - reference_genes)
    fn = len(reference_genes - selected_genes)
    tn = len(all_genes - (selected_genes | reference_genes))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "jaccard": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": specificity,
        "fpr": None if specificity is None else 1.0 - specificity,
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
        "std": _std(values_array),
        "n": int(len(clean_values)),
    }


def _std(values: np.ndarray) -> float:
    """Population std, exactly 0 when all values are equal.

    np.std of identical floats can return float noise (e.g. 1e-16 for ten copies
    of 0.9756...), which would turn a degenerate null into a z-score of ~1e15.
    """
    if np.ptp(values) == 0:
        return 0.0
    return float(np.std(values))


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
    random_std = _std(random_array)
    z_score = (
        (float(observed) - float(np.mean(random_array))) / random_std
        if random_std != 0
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
