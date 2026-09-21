from dataclasses import dataclass
from typing import Literal, TypedDict


EnrichmentSetName = Literal["de_markers", "cell_marker", "sctype"]
BiologicalComparisonRecord = dict[str, object]


@dataclass(frozen=True)
class StratifiedHVGSampleSet:
    """Container for observed HVGs and matched random control samples."""

    observed_hvg_genes: list[str]
    control_samples: list[list[str]]


class BiologicalMetricsRecord(TypedDict):
    """The fixed-shape record `compute_run_metrics` produces for one cluster.

    `attach_tail_statistics` widens this into the looser
    `BiologicalComparisonRecord` by merging in dynamically-named
    significance-stat fields (e.g. ``jaccard_z_score``, ``random_fpr_mean``)
    whose keys depend on which metric they belong to.
    """

    enrichment_set: EnrichmentSetName
    cluster_id: str
    cluster_size: int
    marker_genes: list[str]
    n_marker_genes: int
    cell_type: str
    sample_type: str
    sample_index: int
    sample_seed: int
    run_hvg_count: int
    jaccard: float | None
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    specificity: float | None
    fpr: float | None
    intersection: int
    expected: float | None
    fold_enrichment: float | None


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
