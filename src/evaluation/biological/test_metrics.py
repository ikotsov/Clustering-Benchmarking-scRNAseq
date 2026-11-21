import numpy as np
import pandas as pd
import pytest

from .main import RECORD_COLUMNS
from .metrics import (
    METRIC_NAMES,
    NULL_STATISTIC_NAMES,
    count_overlaps,
    null_statistics,
    overlap_metrics,
    summarize_biological_evaluation,
    untested_statistics,
)


def test_overlap_metrics_match_the_set_formulas():
    # Universe G1..G4; markers {G1, G2}; reference {G1, G3} -> tp=G1, fp=G2, fn=G3, tn=G4
    overlap = count_overlaps({"G1", "G2"}, [{"G1", "G3"}])

    metrics = overlap_metrics(overlap, n_marker_genes=2, reference_sizes=np.array([2]), universe_size=4)

    assert tuple(metrics) == METRIC_NAMES
    assert [int(metrics[name][0]) for name in ("tp", "fp", "fn", "tn")] == [1, 1, 1, 1]
    assert metrics["jaccard"][0] == pytest.approx(1 / 3)
    for name in ("precision", "recall", "specificity", "fpr", "expected", "fold_enrichment"):
        assert metrics[name][0] == pytest.approx(0.5 if name not in ("expected", "fold_enrichment") else 1.0)


def test_null_statistics_test_each_observed_overlap_against_its_column():
    observed = np.array([5, 1])
    random_overlaps = np.array([[1, 1], [2, 3], [5, 0]])

    stats = null_statistics(observed, random_overlaps)

    assert tuple(stats) == NULL_STATISTIC_NAMES
    assert stats["null_overlap_mean"] == pytest.approx([8 / 3, 4 / 3])
    # p = (#random >= observed + 1) / (#random + 1)
    assert stats["p_value"] == pytest.approx([2 / 4, 3 / 4])
    assert stats["overlap_z_score"][0] == pytest.approx((5 - 8 / 3) / np.std([1, 2, 5]))


def test_null_statistics_have_no_z_score_for_a_constant_null():
    stats = null_statistics(np.array([3]), np.full((10, 1), 3))

    assert np.isnan(stats["overlap_z_score"][0])
    assert stats["p_value"][0] == pytest.approx(1.0)


def _records(rows: list[tuple[str, str, float, float | None, str]]) -> pd.DataFrame:
    """rows: (cluster_id, cell_type, jaccard, p_value, null_status)"""
    return pd.DataFrame([
        {"enrichment_set": "de_markers", "cluster_id": cluster_id, "cell_type": cell_type, "jaccard": jaccard,
         "fold_enrichment": 10 * jaccard, "p_value": np.nan if p_value is None else p_value, "null_status": status}
        for cluster_id, cell_type, jaccard, p_value, status in rows
    ])


def test_summarize_biological_evaluation_uses_each_cell_types_best_cluster():
    records = _records([
        ("0", "A", 0.8, 0.01, "tested"),
        ("1", "A", 0.2, 0.01, "tested"),            # a weaker piece of A
        ("1", "B", 0.3, 0.50, "tested"),            # best for B, but not significant
        ("2", "B", 0.1, None, "too_few_markers"),
    ])
    reference_sets = {"de_markers": {"A": {"g1"}, "B": {"g2"}, "C": {"g3"}}}  # no cluster has markers for C

    summary = summarize_biological_evaluation(records, reference_sets, n_clusters=4)

    assert summary["bio_de_markers_jaccard"] == pytest.approx((0.8 + 0.3 + 0.0) / 3)
    assert summary["bio_de_markers_fold_enrichment"] == pytest.approx((8 + 3 + 0) / 3)
    assert summary["bio_de_markers_n_recovered"] == 1
    assert summary["bio_de_markers_n_cell_types"] == 3
    assert summary["bio_n_clusters"] == 4
    assert summary["bio_n_clusters_without_markers"] == 1  # cluster 3 has no records
    assert summary["bio_n_clusters_untested"] == 1


def test_summarize_biological_evaluation_scores_a_run_without_markers_as_zero():
    records = pd.DataFrame(columns=RECORD_COLUMNS)

    summary = summarize_biological_evaluation(records, {"de_markers": {"A": {"g1"}}}, n_clusters=1)

    assert summary["bio_de_markers_jaccard"] == 0.0
    assert summary["bio_de_markers_n_recovered"] == 0
    assert summary["bio_n_clusters_without_markers"] == 1


def test_untested_statistics_fill_every_null_statistic_column():
    assert tuple(untested_statistics()) == NULL_STATISTIC_NAMES
