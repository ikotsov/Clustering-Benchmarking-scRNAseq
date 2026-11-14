from typing import get_args

from src.tuning.types import ExternalObjectiveMetric, InternalObjectiveMetric, ObjectiveMetric, TuningObjective


def get_tuning_objective(objective_metric: ObjectiveMetric) -> TuningObjective:
    """Whether the metric scores against the ground truth labels (external) or not (internal)."""
    if objective_metric in get_args(ExternalObjectiveMetric):
        return "external"
    if objective_metric in get_args(InternalObjectiveMetric):
        return "internal"
    raise ValueError(f"Unknown objective metric: {objective_metric}")


def get_params_label(tuning_metric: ObjectiveMetric | None) -> str:
    """
    The {params} label of the tuning and run paths (see src.paths): "default" for a run with the algorithm's
    defaults, else "tuned_<metric>" for parameters tuned on that metric (e.g. "tuned_silhouette").
    """
    return "default" if tuning_metric is None else f"tuned_{tuning_metric}"
