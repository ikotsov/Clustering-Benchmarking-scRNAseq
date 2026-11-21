import json

from .io import save_biological_evaluation_results


def test_save_biological_evaluation_results_keeps_default_and_tuned_runs_apart(tmp_path):
    def save(tuning_metric, records):
        save_biological_evaluation_results(
            dataset="D", algorithm="leiden", preprocessing="log_cpm", with_pca=True,
            tuning_metric=tuning_metric, comparison_records=records, output_dir=str(tmp_path),
        )

    save(None, [{"jaccard": 0.1}])
    save("silhouette", [{"jaccard": 0.2}])
    save(None, [{"jaccard": 0.3}])  # replaces the first default entry, not the tuned one

    runs = json.loads((tmp_path / "biological_results.json").read_text())["runs"]
    by_metric = {run["tuning_metric"]: run["comparison_records"] for run in runs}
    assert len(runs) == 2
    assert by_metric == {None: [{"jaccard": 0.3}], "silhouette": [{"jaccard": 0.2}]}
