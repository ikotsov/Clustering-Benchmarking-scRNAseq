import json
import os
from datetime import datetime

from src.utils import get_pca_label


def save_evaluation_results(
    dataset: str,
    algorithm: str,
    preprocessing: str,
    with_pca: bool,
    metrics: dict[str, float],
    output_dir: str,
) -> None:
    """
    Save evaluation results into a shared results.json file.

    Results are stored in a flat runs structure:
        {
          "dataset": "...",
          "updated_at": "...",
          "runs": [
            {
              "algorithm": "kmeans",
              "normalization": "pearson",
              "with_pca": true,
              "metrics": {...},
              "timestamp": "..."
            }
          ]
        }

    Re-running an experiment overwrites the matching run entry
    (algorithm + normalization + with_pca).

    Parameters
    ----------
    dataset : str
        Dataset accession ID
    algorithm : str
        Clustering algorithm name
    preprocessing : str
        Normalization method used (e.g. "log_cpm", "pearson")
    with_pca : bool
        Whether PCA was applied
    metrics : dict
        Dictionary of metric names and values (e.g., {"ari": 0.85, "nmi": 0.78})
    output_dir : str
        Directory containing results.json
    """
    results_file = os.path.join(output_dir, "results.json")
    now_iso = datetime.now().isoformat()

    all_results: dict[str, object]
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            loaded = json.load(f)

        all_results = loaded
    else:
        all_results = {
            "dataset": dataset,
            "updated_at": now_iso,
            "runs": [],
        }

    runs = all_results["runs"]
    if not isinstance(runs, list):
        raise ValueError("Invalid results.json format: 'runs' must be a list.")

    run_entry = {
        "algorithm": algorithm,
        "normalization": preprocessing,
        "with_pca": with_pca,
        "metrics": metrics,
        "timestamp": now_iso,
    }

    replaced = False
    for idx, existing in enumerate(runs):
        if not isinstance(existing, dict):
            continue

        if (
            existing.get("algorithm") == algorithm
            and existing.get("normalization") == preprocessing
            and bool(existing.get("with_pca", False)) == with_pca
        ):
            runs[idx] = run_entry
            replaced = True
            break

    if not replaced:
        runs.append(run_entry)

    all_results.setdefault("dataset", dataset)
    all_results["updated_at"] = now_iso

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    pca_key = get_pca_label(with_pca)
    print(
        f"  ✓ Evaluation saved: {algorithm} → {preprocessing} → {pca_key} in results.json")
