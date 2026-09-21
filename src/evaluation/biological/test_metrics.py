import numpy as np
import pytest

from .metrics import _build_null_pools, attach_tail_statistics, compute_run_metrics


def test_compute_run_metrics_returns_raw_values_without_tail_stats():
    # universe: G1, G2, G3, G4. Cluster picked G1, G2. Reference "B_cells" is G1, G3.
    # -> tp=G1, fp=G2, fn=G3, tn=G4.
    record = compute_run_metrics(
        cluster_marker_genes=["G1", "G2"],
        enrichment_sets={"de_markers": {"B_cells": {"G1", "G3"}}},
        all_genes={"G1", "G2", "G3", "G4"},
        cluster_id="0",
        cluster_size=10,
        sample_type="observed",
        sample_index=0,
        sample_seed=3,
        run_hvg_count=2,
    )[0]

    assert record["tp"] == 1
    assert record["fp"] == 1
    assert record["fn"] == 1
    assert record["tn"] == 1
    assert record["jaccard"] == pytest.approx(1 / 3)
    assert record["precision"] == pytest.approx(0.5)
    assert record["recall"] == pytest.approx(0.5)
    assert record["specificity"] == pytest.approx(0.5)
    assert record["fpr"] == pytest.approx(0.5)
    assert record["intersection"] == 1
    assert record["expected"] == pytest.approx(1.0)
    assert record["fold_enrichment"] == pytest.approx(1.0)

    # Raw metrics only -- no significance stats yet (those come from attach_tail_statistics).
    assert "jaccard_z_score" not in record
    assert "random_jaccard_mean" not in record


def _make_record(sample_index: int, jaccard: float, fpr: float) -> dict:
    """Build a minimal, fully-populated comparison record for null-pooling tests."""
    return {
        "enrichment_set": "de_markers",
        "cell_type": "B_cells",
        "sample_index": sample_index,
        "jaccard": jaccard,
        "precision": jaccard,
        "recall": jaccard,
        "specificity": jaccard,
        "fpr": fpr,
        "fold_enrichment": jaccard,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "intersection": 1,
        "expected": 1.0,
    }


def test_build_null_pools_takes_max_per_rerun_for_right_tail_metrics_and_min_for_fpr():
    # Two random reruns, each contributing two clusters (mimicking two
    # clustering runs that each produced a different number of clusters).
    sampled_records = [
        _make_record(sample_index=1, jaccard=0.05, fpr=0.30),
        _make_record(sample_index=1, jaccard=0.08, fpr=0.20),
        _make_record(sample_index=2, jaccard=0.03, fpr=0.25),
        _make_record(sample_index=2, jaccard=0.10, fpr=0.15),
    ]

    pools = _build_null_pools(sampled_records)
    metric_pools = pools[("de_markers", "B_cells")]

    # One null value per rerun (2 reruns), taking each rerun's best cluster.
    assert metric_pools["jaccard"] == pytest.approx([0.08, 0.10])
    # ...and each rerun's *lowest* fpr, since low fpr is the meaningful direction.
    assert metric_pools["fpr"] == pytest.approx([0.20, 0.15])


def test_attach_tail_statistics_populates_real_stats_from_sampled_runs():
    observed_records = [
        {
            "enrichment_set": "de_markers",
            "cell_type": "B_cells",
            "cluster_id": "0",
            "sample_index": 0,
            "jaccard": 0.6,
            "precision": 0.6,
            "recall": 0.6,
            "specificity": 0.6,
            "fpr": 0.1,
            "fold_enrichment": 0.6,
            "tp": 5,
            "fp": 2,
            "fn": 1,
            "tn": 10,
            "intersection": 5,
            "expected": 2.0,
        }
    ]
    sampled_records = [
        _make_record(sample_index=1, jaccard=0.05, fpr=0.30),
        _make_record(sample_index=1, jaccard=0.08, fpr=0.20),
        _make_record(sample_index=2, jaccard=0.03, fpr=0.25),
        _make_record(sample_index=2, jaccard=0.10, fpr=0.15),
    ]

    [enriched] = attach_tail_statistics(observed_records, sampled_records)

    # This is the regression check for the bug: these used to always be None
    # because the null distribution was a hardcoded empty list.
    assert enriched["jaccard_z_score"] is not None
    assert enriched["jaccard_p_value"] is not None
    assert enriched["jaccard_ci_lower"] is not None
    assert enriched["jaccard_ci_upper"] is not None

    null_jaccard = [0.08, 0.10]
    expected_z = (0.6 - np.mean(null_jaccard)) / np.std(null_jaccard)
    assert enriched["jaccard_z_score"] == pytest.approx(expected_z)
    assert enriched["random_jaccard_mean"] == pytest.approx(np.mean(null_jaccard))
    assert enriched["random_jaccard_n"] == 2

    # fpr is a left-tail metric: observed (0.1) is lower than both pooled
    # random minimums (0.20, 0.15), so it should look significant (small p).
    assert enriched["random_fpr_mean"] == pytest.approx(np.mean([0.20, 0.15]))
    assert enriched["fpr_p_value"] == pytest.approx(1 / 3)


def test_attach_tail_statistics_handles_no_sampled_runs():
    observed_records = [
        {
            "enrichment_set": "de_markers",
            "cell_type": "B_cells",
            "cluster_id": "0",
            "sample_index": 0,
            "jaccard": 0.6,
            "precision": 0.6,
            "recall": 0.6,
            "specificity": 0.6,
            "fpr": 0.1,
            "fold_enrichment": 0.6,
            "tp": 5,
            "fp": 2,
            "fn": 1,
            "tn": 10,
            "intersection": 5,
            "expected": 2.0,
        }
    ]

    [enriched] = attach_tail_statistics(observed_records, sampled_records=[])

    assert enriched["jaccard_z_score"] is None
    assert enriched["jaccard_p_value"] is None
    assert enriched["random_jaccard_n"] == 0
