from typing_extensions import Literal

from src.types import ExternalMetric, InternalMetric

TuningObjective = Literal["external", "internal"]
ExternalObjectiveMetric = Literal[ExternalMetric, "mean_external_score"]
InternalObjectiveMetric = InternalMetric
ObjectiveMetric = Literal[ExternalObjectiveMetric, InternalObjectiveMetric]
