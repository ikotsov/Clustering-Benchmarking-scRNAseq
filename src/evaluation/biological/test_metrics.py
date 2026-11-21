import numpy as np
import pytest

from .metrics import _build_null_pools, attach_tail_statistics, compute_run_metrics, summarize_biological_evaluation


def test_compute_run_metrics_returns_raw_values_without_tail_stats():
    # universe: G1, G2, G3, G4. Cluster picked G1, G2. Reference "B_cells" is G1, G3.
    # -> tp=G1, fp=G2, fn=G3, tn=G4.
    record = compute_run_metrics(
        cluster_marker_genes=["G1", "G2"],
        enrichment_sets={"de_markers": {"B_cells": {"G1", "G3"}}},
        all_genes={"G1", "G2", "G3", "G4"},
        cluster_id="0",
        cluster_size=10,
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


def _make_record(sample_index: int, jaccard: float, fpr: float, cluster_id: str = "0") -> dict:
    """Build a minimal, fully-populated comparison record for null-pooling tests."""
    return {
        "enrichment_set": "de_markers",
        "cluster_id": cluster_id,
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


def test_build_null_pools_keeps_one_value_per_random_set_per_cluster():
    sampled_records = [
        _make_record(sample_index=1, jaccard=0.05, fpr=0.30, cluster_id="0"),
        _make_record(sample_index=2, jaccard=0.08, fpr=0.20, cluster_id="0"),
        _make_record(sample_index=1, jaccard=0.03, fpr=0.25, cluster_id="1"),
        _make_record(sample_index=2, jaccard=0.10, fpr=0.15, cluster_id="1"),
    ]

    pools = _build_null_pools(sampled_records)

    # No pooling across clusters: each cluster is tested against its own random sets.
    assert pools[("de_markers", "0", "B_cells")]["jaccard"] == pytest.approx([0.05, 0.08])
    assert pools[("de_markers", "1", "B_cells")]["jaccard"] == pytest.approx([0.03, 0.10])
    assert pools[("de_markers", "1", "B_cells")]["fpr"] == pytest.approx([0.25, 0.15])


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
        _make_record(sample_index=1, jaccard=0.08, fpr=0.20),
        _make_record(sample_index=2, jaccard=0.10, fpr=0.15),
        # Another cluster's random sets must not enter cluster "0"'s null.
        _make_record(sample_index=1, jaccard=0.90, fpr=0.01, cluster_id="1"),
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

    # fpr is a left-tail metric: observed (0.1) is lower than both random
    # values (0.20, 0.15), so it should look significant (small p).
    assert enriched["random_fpr_mean"] == pytest.approx(np.mean([0.20, 0.15]))
    assert enriched["fpr_p_value"] == pytest.approx(1 / 3)


def test_attach_tail_statistics_has_no_z_score_when_the_null_is_constant():
    # Identical null values whose np.std is float noise (~1e-16), not 0.
    value = 40 / 41
    observed = _make_record(sample_index=0, jaccard=0.5, fpr=0.1)
    observed["cluster_id"] = "0"
    sampled = [_make_record(sample_index=i, jaccard=value, fpr=0.2) for i in range(1, 11)]
    assert np.std([value] * 10) != 0  # the float noise this test guards against

    [enriched] = attach_tail_statistics([observed], sampled)

    assert enriched["jaccard_z_score"] is None
    assert enriched["random_jaccard_std"] == 0.0
    assert enriched["jaccard_p_value"] == pytest.approx(1.0)


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


def _summary_record(cluster_id: str, cell_type: str, jaccard: float, p_value: float | None) -> dict:
    return {"enrichment_set": "de_markers", "cluster_id": cluster_id, "cell_type": cell_type,
            "jaccard": jaccard, "fold_enrichment": 10 * jaccard, "jaccard_p_value": p_value}


def test_summarize_biological_evaluation_uses_each_cell_types_best_cluster():
    records = [
        _summary_record("0", "A", jaccard=0.8, p_value=0.01),
        _summary_record("1", "A", jaccard=0.2, p_value=0.01),   # a weaker piece of A
        _summary_record("1", "B", jaccard=0.3, p_value=0.50),   # best for B, but not significant
        _summary_record("2", "B", jaccard=0.1, p_value=None),   # untested cluster
    ]
    reference_sets = {"de_markers": {"A": {"g1"}, "B": {"g2"}, "C": {"g3"}}}  # no cluster has markers for C

    summary = summarize_biological_evaluation(
        records, reference_sets, n_clusters=4, n_clusters_without_markers=1, n_clusters_untested=1)

    assert summary["bio_de_markers_jaccard"] == pytest.approx((0.8 + 0.3 + 0.0) / 3)
    assert summary["bio_de_markers_fold_enrichment"] == pytest.approx((8 + 3 + 0) / 3)
    assert summary["bio_de_markers_n_recovered"] == 1
    assert summary["bio_de_markers_n_cell_types"] == 3
    assert (summary["bio_n_clusters"], summary["bio_n_clusters_without_markers"], summary["bio_n_clusters_untested"]) == (4, 1, 1)
