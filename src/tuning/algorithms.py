import logging
import math
import statistics
from collections.abc import Callable
from typing import Literal, TypedDict, cast

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import pairwise_distances

from src.clustering.registry import (
    ALGORITHMS_REQUIRING_N_CLUSTERS,
    ClusteringAlgorithm,
    get_clustering_strategy,
)
from src.constants import SEED
from src.data_loading import load_ground_truth_labels
from src.dataset_config import load_dataset_config, parse_n_clusters
from src.evaluation.extrinsic import score_external
from src.evaluation.intrinsic import LOWER_IS_BETTER, score_internal
from src.paths import get_pca_label
from src.tuning.common import (
    DEFAULT_NORM_METHOD,
    ExternalObjectiveMetric,
    InternalObjectiveMetric,
    ObjectiveMetric,
    get_tuning_objective,
    load_preprocessed_data,
    save_tuning_results,
)
from src.tuning.types import TuningObjective
from src.types import NormMethod, PathLike


logger = logging.getLogger(__name__)


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
    "hdbscan": {
        "min_cluster_size": {"kind": "int", "low": 3, "high": 30},
        "min_samples": {"kind": "int", "low": 1, "high": 10},
    },
    "optics": {
        "min_samples": {"kind": "int", "low": 3, "high": 20},
    },
}


def run_tuning(
    algorithm: ClusteringAlgorithm,
    *,
    data_path: PathLike,
    config_path: PathLike,
    labels_path: PathLike | None,
    output_path: PathLike,
    n_trials: int = 50,
    norm_method: NormMethod = DEFAULT_NORM_METHOD,
    with_pca: bool = True,
    objective: TuningObjective = 'external',
    objective_metric: ObjectiveMetric = 'mean_external_score',
) -> None:
    """
    Tunes the parameters of one algorithm with Optuna and saves the best ones (see save_tuning_results).

    External objective metrics score each trial against the ground truth labels (labels_path);
    internal ones score the clusters on the data alone, as a user without labels would tune,
    so labels_path is not needed. objective_metric must belong to objective.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if algorithm not in ALGORITHM_PARAM_SPECS:
        supported = ", ".join(sorted(ALGORITHM_PARAM_SPECS.keys()))
        raise ValueError(
            f"Tuning is not configured for '{algorithm}'. Supported: {supported}")

    if get_tuning_objective(objective_metric) != objective:
        raise ValueError(f"'{objective_metric}' is not an {objective} metric")

    data = load_preprocessed_data(data_path)

    score: Callable[[pd.Series], float]
    if objective == "external":
        if labels_path is None:
            raise ValueError(
                f"Tuning on '{objective_metric}' needs the ground truth labels (labels_path)")
        score = _external_scorer(
            load_ground_truth_labels(labels_path), cast(ExternalObjectiveMetric, objective_metric))
    else:
        score = _internal_scorer(data, cast(InternalObjectiveMetric, objective_metric))

    fixed_params: dict[str, int] = {}
    n_clusters = parse_n_clusters(load_dataset_config(config_path))
    if algorithm in ALGORITHMS_REQUIRING_N_CLUSTERS and n_clusters is not None:
        fixed_params["n_clusters"] = n_clusters

    study = optuna.create_study(
        direction="minimize" if objective_metric in LOWER_IS_BETTER else "maximize",
        study_name=f"{algorithm}_{norm_method}_{get_pca_label(with_pca)}_{objective}_tuning",
        sampler=TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: _objective(trial, data, algorithm, fixed_params, score),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    best_value = float(
        best_trial.value) if best_trial.value is not None else float("nan")
    best_params = ", ".join(f"{k}={v}" for k, v in best_trial.params.items())

    logger.info("Tuning complete (%s). Best params: %s",
                algorithm, best_params)
    logger.info("Best %s: %.4f", objective_metric, best_value)
    if math.isinf(best_value):
        logger.warning(
            "%s was undefined in every trial (fewer than 2 clusters, or only singletons);"
            " the saved parameters are arbitrary", objective_metric)

    save_tuning_results(
        output_path=output_path,
        algorithm=algorithm,
        norm_method=norm_method,
        with_pca=with_pca,
        best_params=best_trial.params,
        best_value=best_value,
        objective_metric=objective_metric,
    )


def _objective(
    trial: optuna.Trial,
    data: pd.DataFrame,
    algorithm: ClusteringAlgorithm,
    fixed_params: dict[str, int],
    score: Callable[[pd.Series], float],
) -> float:
    param_specs = ALGORITHM_PARAM_SPECS[algorithm]
    params = _suggest_params(trial, param_specs)

    strategy = get_clustering_strategy(algorithm)
    labels_pred = strategy(data, **params, **fixed_params)
    return score(labels_pred)


def _external_scorer(
    labels_true: pd.Series,
    objective_metric: ExternalObjectiveMetric,
) -> Callable[[pd.Series], float]:
    def score(labels_pred: pd.Series) -> float:
        common_cells = labels_pred.index.intersection(labels_true.index)
        scores = score_external(labels_true.loc[common_cells], labels_pred.loc[common_cells])
        if objective_metric == "mean_external_score":
            return statistics.fmean(scores.values())
        return scores[objective_metric]

    return score


def _internal_scorer(
    data: pd.DataFrame,
    objective_metric: InternalObjectiveMetric,
) -> Callable[[pd.Series], float]:
    X = data.to_numpy()
    # score_internal always computes silhouette, which needs every pairwise distance;
    # computing them once saves recomputing them for each trial.
    distances = pairwise_distances(X)
    # Trials whose labels leave the metric undefined get its worst value, steering the search away from them.
    worst = math.inf if objective_metric in LOWER_IS_BETTER else -math.inf

    def score(labels_pred: pd.Series) -> float:
        value = score_internal(
            X, labels_pred.loc[data.index].to_numpy(), distances)[objective_metric]
        return worst if value is None else float(value)

    return score


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
