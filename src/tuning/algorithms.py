import os
from typing import Literal, TypedDict

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.clustering.registry import ClusteringAlgorithm, get_clustering_strategy
from src.constants import SEED
from src.data_loading import load_ground_truth_labels
from src.tuning.common import (
    ObjectiveMetric,
    compute_clustering_metrics,
    load_preprocessed_data,
    validate_objective_metric,
)


class ParamSpec(TypedDict):
    kind: Literal["float", "int"]
    low: float
    high: float


ALGORITHM_PARAM_SPECS: dict[ClusteringAlgorithm, dict[str, ParamSpec]] = {
    "leiden": {
        "resolution": {"kind": "float", "low": 0.1, "high": 3.0},
    },
    "birch": {
        "threshold": {"kind": "float", "low": 0.1, "high": 2.0},
        "branching_factor": {"kind": "int", "low": 20, "high": 100},
    },
}


def run_tuning(
    accession: str,
    algorithm: ClusteringAlgorithm,
    n_trials: int = 50,
    objective_metric: ObjectiveMetric = 'mean_external_score',
) -> None:
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if algorithm not in ALGORITHM_PARAM_SPECS:
        supported = ", ".join(sorted(ALGORITHM_PARAM_SPECS.keys()))
        raise ValueError(
            f"Tuning is not configured for '{algorithm}'. Supported: {supported}")

    objective_metric = validate_objective_metric(objective_metric)

    data = load_preprocessed_data(accession)

    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(project_root, "data", accession)
    labels_true = load_ground_truth_labels(dataset_dir)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{accession}_{algorithm}_tuning",
        sampler=TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: _objective(
            trial, data, labels_true, algorithm, objective_metric),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    best_value = float(
        best_trial.value) if best_trial.value is not None else float("nan")
    best_params = ", ".join(f"{k}={v}" for k, v in best_trial.params.items())

    print(f"Tuning complete ({algorithm}). Best params: {best_params}")
    print(f"Best {objective_metric}: {best_value:.4f}")


def _objective(
    trial: optuna.Trial,
    data: pd.DataFrame,
    labels_true: pd.Series,
    algorithm: ClusteringAlgorithm,
    objective_metric: ObjectiveMetric,
) -> float:
    param_specs = ALGORITHM_PARAM_SPECS[algorithm]
    params = _suggest_params(trial, param_specs)

    strategy = get_clustering_strategy(algorithm)
    labels_pred = strategy(data, **params)

    common_cells = labels_pred.index.intersection(labels_true.index)
    y_pred = labels_pred.loc[common_cells]
    y_true = labels_true.loc[common_cells]

    metrics = compute_clustering_metrics(y_pred, y_true)
    return float(metrics[objective_metric])


def _suggest_params(
    trial: optuna.Trial,
    param_specs: dict[str, ParamSpec],
) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for param_name, spec in param_specs.items():
        if spec["kind"] == "float":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"])
        else:
            params[param_name] = trial.suggest_int(
                param_name, int(spec["low"]), int(spec["high"]))
    return params
