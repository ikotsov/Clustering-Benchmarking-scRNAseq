import pandas as pd
import pytest

from .biological.io import save_biological_evaluation
from .biological.types import BiologicalEvaluation
from .io import aggregate_run_results, save_run_result


def _save_run(path, algorithm):
    save_run_result(
        algorithm=algorithm, preprocessing="log_cpm", with_pca=True, tuning_metric=None,
        metrics={"ari": 0.5}, output_path=path, n_cells=10, n_features=2,
        clustering_resources={"wall_time_s": 1.0},
    )


def test_aggregate_run_results_adds_the_biological_summary_of_each_run(tmp_path):
    runs, biologicals = [], []
    for algorithm, jaccard in [("kmeans", 0.4), ("leiden", 0.9)]:
        runs.append(tmp_path / f"{algorithm}.json")
        biologicals.append(tmp_path / f"{algorithm}_biological.json")
        _save_run(runs[-1], algorithm)
        save_biological_evaluation(
            biologicals[-1],
            BiologicalEvaluation(records=[], summary={"bio_de_markers_jaccard": jaccard, "bio_n_clusters": 3}),
            n_samples=100, seed=3,
        )

    aggregate_run_results("D", runs, tmp_path / "results.tsv", biological_result_paths=biologicals)

    table = pd.read_csv(tmp_path / "results.tsv", sep="\t").set_index("algorithm")
    assert table.loc["kmeans", "bio_de_markers_jaccard"] == pytest.approx(0.4)
    assert table.loc["leiden", "bio_de_markers_jaccard"] == pytest.approx(0.9)
    assert table.loc["leiden", "bio_n_clusters"] == 3


def test_aggregate_run_results_needs_one_biological_result_per_run(tmp_path):
    _save_run(tmp_path / "kmeans.json", "kmeans")

    with pytest.raises(ValueError, match="biological"):
        aggregate_run_results("D", [tmp_path / "kmeans.json"], tmp_path / "results.tsv", biological_result_paths=[])
