import pytest

from .main import evaluate_clustering_biologically
from .markers import build_reference_sets
from .test_markers import _synthetic_counts


def test_evaluate_clustering_biologically_recovers_the_cell_types_of_a_perfect_clustering():
    counts, labels = _synthetic_counts()
    reference_sets = build_reference_sets(counts, labels, species="human")

    evaluation = evaluate_clustering_biologically(counts, labels, reference_sets, n_samples=20, seed=3)

    summary = evaluation.summary
    assert summary["bio_de_markers_jaccard"] == pytest.approx(1.0)
    assert summary["bio_de_markers_n_recovered"] == 3
    assert summary["bio_n_clusters_untested"] == 0
    matching = [r for r in evaluation.records if r["cluster_id"] == r["cell_type"]]
    assert all(r["null_status"] == "tested" and r["jaccard_p_value"] == pytest.approx(1 / 21) for r in matching)


def test_evaluate_clustering_biologically_does_not_test_clusters_with_too_few_markers():
    counts, labels = _synthetic_counts()
    reference_sets = build_reference_sets(counts, labels, species="human")

    evaluation = evaluate_clustering_biologically(
        counts, labels, reference_sets, n_samples=20, seed=3, min_marker_genes=1000)

    assert {r["null_status"] for r in evaluation.records} == {"too_few_markers"}
    assert all(r["jaccard_p_value"] is None for r in evaluation.records)
    assert evaluation.summary["bio_de_markers_n_recovered"] == 0
    assert evaluation.summary["bio_n_clusters_untested"] == 3
