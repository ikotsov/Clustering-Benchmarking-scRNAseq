import pytest

from .main import evaluate_clustering_biologically
from .markers import build_reference_sets
from .test_markers import _synthetic_counts
from .types import NullSettings


def test_evaluate_clustering_biologically_recovers_the_cell_types_of_a_perfect_clustering():
    counts, labels = _synthetic_counts()
    reference_sets = build_reference_sets(counts, labels, species="human")

    evaluation = evaluate_clustering_biologically(counts, labels, reference_sets, NullSettings(n_samples=20))

    assert evaluation.summary["bio_de_markers_jaccard"] == pytest.approx(1.0)
    assert evaluation.summary["bio_de_markers_n_recovered"] == 3
    assert evaluation.summary["bio_n_clusters_untested"] == 0
    matching = evaluation.records[evaluation.records["cluster_id"] == evaluation.records["cell_type"]]
    assert (matching["null_status"] == "tested").all()
    assert matching["p_value"].tolist() == pytest.approx([1 / 21] * 3)
    assert set(evaluation.cluster_markers) == {"a", "b", "c"}


def test_evaluate_clustering_biologically_does_not_test_clusters_with_too_few_markers():
    counts, labels = _synthetic_counts()
    reference_sets = build_reference_sets(counts, labels, species="human")

    evaluation = evaluate_clustering_biologically(
        counts, labels, reference_sets, NullSettings(n_samples=20, min_marker_genes=1000))

    assert set(evaluation.records["null_status"]) == {"too_few_markers"}
    assert evaluation.records["p_value"].isna().all()
    assert evaluation.summary["bio_de_markers_n_recovered"] == 0
    assert evaluation.summary["bio_n_clusters_untested"] == 3


def test_evaluate_clustering_biologically_scores_a_single_cluster_as_zero():
    counts, labels = _synthetic_counts()
    reference_sets = build_reference_sets(counts, labels, species="human")
    one_cluster = labels.map(lambda _: "0")

    evaluation = evaluate_clustering_biologically(counts, one_cluster, reference_sets, NullSettings(n_samples=20))

    assert evaluation.records.empty
    assert evaluation.summary["bio_de_markers_jaccard"] == 0.0
    assert evaluation.summary["bio_n_clusters_without_markers"] == 1


def test_evaluate_clustering_biologically_rejects_cells_outside_the_counts():
    counts, labels = _synthetic_counts()
    reference_sets = build_reference_sets(counts, labels, species="human")

    with pytest.raises(ValueError, match="same cell QC"):
        evaluate_clustering_biologically(counts.iloc[1:], labels, reference_sets)
