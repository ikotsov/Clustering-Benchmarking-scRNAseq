from dataclasses import dataclass
from typing import Literal, TypedDict


EnrichmentSetName = Literal["de_markers", "cell_marker", "sctype"]
BiologicalComparisonRecord = dict[str, object]


@dataclass(frozen=True)
class StratifiedHVGSampleSet:
    """Container for observed HVGs and matched random control samples."""

    observed_hvg_genes: list[str]
    control_samples: list[list[str]]


class MembershipMetrics(TypedDict):
    tp: int
    fp: int
    fn: int
    tn: int
    jaccard: float | None
    precision: float | None
    recall: float | None
    specificity: float | None
    fpr: float | None


class FoldEnrichmentMetrics(TypedDict):
    intersection: int
    expected: float | None
    fold_enrichment: float | None


class TailStats(TypedDict):
    z_score: float | None
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None


class RandomSummaryStats(TypedDict):
    mean: float | None
    std: float | None
    n: int
