from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score

from .types import (
    BiologicalComparisonRecord,
    EnrichmentSetName,
    FoldEnrichmentMetrics,
    MembershipMetrics,
    RandomSummaryStats,
    StratifiedHVGSampleSet,
    TailStats,
)


def compute_biological_metrics_results(
    stratified_hvg_samples: StratifiedHVGSampleSet,
    enrichment_sets: dict[EnrichmentSetName, dict[str, set[str]]],
    all_genes: set[str],
    cluster_id: str,
    cluster_size: int,
    cluster_marker_genes: list[str],
) -> list[BiologicalComparisonRecord]:
    """Compute per-comparison biological metrics using observed vs random HVGs.

    Each returned record corresponds to one predicted-cluster marker set compared
    against one reference cell type marker set (within an enrichment set), with
    cluster metadata attached for persistence.
    """
    observed_hvgs = set(stratified_hvg_samples.observed_hvg_genes)
    random_hvg_samples = [set(sample)
                          for sample in stratified_hvg_samples.control_samples]

    universe = sorted(all_genes)
    observed_hvg_vector = _binary_membership_vector(universe, observed_hvgs)
    random_hvg_vectors = [
        _binary_membership_vector(universe, random_sample)
        for random_sample in random_hvg_samples
    ]

    results: list[BiologicalComparisonRecord] = []

    # for each reference enrichment set and cell type, compare observed and random HVG sets to the reference marker genes.
    for enrichment_set_name, marker_sets_by_cell_type in enrichment_sets.items():
        # for each cell type in the reference enrichment set, compare observed and random HVG sets to the reference marker genes.
        for cell_type, reference_marker_genes in marker_sets_by_cell_type.items():
            marker_set = set(reference_marker_genes)
            y_true = _binary_membership_vector(universe, marker_set)

            observed_metrics = _compute_membership_metrics(
                y_true=y_true,
                y_pred=observed_hvg_vector,
            )
            observed_fold = _compute_fold_enrichment(
                selected_genes=observed_hvgs,
                marker_genes=marker_set,
                all_genes=all_genes,
            )

            random_metrics_for_cell_type: list[MembershipMetrics] = []
            random_fold_values: list[float | None] = []
            random_fold_intersections: list[int] = []
            random_fold_expected: list[float | None] = []

            # for each random sample, compute the same metrics to build a null distribution for comparison.
            for random_sample, random_hvg_vector in zip(
                random_hvg_samples,
                random_hvg_vectors,
            ):
                sample_metrics = _compute_membership_metrics(
                    y_true=y_true,
                    y_pred=random_hvg_vector,
                )
                sample_fold = _compute_fold_enrichment(
                    selected_genes=random_sample,
                    marker_genes=marker_set,
                    all_genes=all_genes,
                )

                random_metrics_for_cell_type.append(sample_metrics)
                random_fold_values.append(sample_fold["fold_enrichment"])
                random_fold_intersections.append(sample_fold["intersection"])
                random_fold_expected.append(sample_fold["expected"])

            jaccard_stats = _right_tail_stats(
                observed=observed_metrics["jaccard"],
                random_values=[metrics["jaccard"]
                               for metrics in random_metrics_for_cell_type],
            )
            precision_stats = _right_tail_stats(
                observed=observed_metrics["precision"],
                random_values=[metrics["precision"]
                               for metrics in random_metrics_for_cell_type],
            )
            recall_stats = _right_tail_stats(
                observed=observed_metrics["recall"],
                random_values=[metrics["recall"]
                               for metrics in random_metrics_for_cell_type],
            )
            specificity_stats = _right_tail_stats(
                observed=observed_metrics["specificity"],
                random_values=[metrics["specificity"]
                               for metrics in random_metrics_for_cell_type],
            )
            fpr_stats = _left_tail_stats(
                observed=observed_metrics["fpr"],
                random_values=[metrics["fpr"]
                               for metrics in random_metrics_for_cell_type],
            )
            fold_stats = _right_tail_stats(
                observed=observed_fold["fold_enrichment"],
                random_values=random_fold_values,
            )

            results.append(
                {
                    "enrichment_set": enrichment_set_name,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "marker_genes": cluster_marker_genes,
                    "n_marker_genes": len(cluster_marker_genes),
                    "cell_type": cell_type,
                    "jaccard": observed_metrics["jaccard"],
                    **_prefix_stats(jaccard_stats, "jaccard"),
                    **_random_summary(
                        "jaccard",
                        [metrics["jaccard"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    "tp": observed_metrics["tp"],
                    "fp": observed_metrics["fp"],
                    "fn": observed_metrics["fn"],
                    "tn": observed_metrics["tn"],
                    "precision": observed_metrics["precision"],
                    "recall": observed_metrics["recall"],
                    "specificity": observed_metrics["specificity"],
                    "fpr": observed_metrics["fpr"],
                    **_prefix_stats(precision_stats, "precision"),
                    **_prefix_stats(recall_stats, "recall"),
                    **_prefix_stats(specificity_stats, "specificity"),
                    **_prefix_stats(fpr_stats, "fpr"),
                    **_random_summary(
                        "precision",
                        [metrics["precision"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    **_random_summary(
                        "recall",
                        [metrics["recall"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    **_random_summary(
                        "specificity",
                        [metrics["specificity"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    **_random_summary(
                        "fpr",
                        [metrics["fpr"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    **_random_summary(
                        "tp",
                        [metrics["tp"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    **_random_summary(
                        "fp",
                        [metrics["fp"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    **_random_summary(
                        "fn",
                        [metrics["fn"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    **_random_summary(
                        "tn",
                        [metrics["tn"]
                            for metrics in random_metrics_for_cell_type],
                    ),
                    "intersection": observed_fold["intersection"],
                    "expected": observed_fold["expected"],
                    "fold_enrichment": observed_fold["fold_enrichment"],
                    **_prefix_stats(fold_stats, "fold_enrichment"),
                    **_random_summary("fold_enrichment", random_fold_values),
                    **_random_summary("intersection", random_fold_intersections),
                    **_random_summary("expected", random_fold_expected),
                }
            )

    return results


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
    clean_values = [float(value) for value in values if pd.notna(value)]
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
    clean_random = [float(value) for value in random_values if pd.notna(value)]
    if pd.isna(observed) or len(clean_random) == 0:
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
