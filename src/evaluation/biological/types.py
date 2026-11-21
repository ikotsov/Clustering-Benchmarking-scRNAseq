from dataclasses import dataclass
from typing import Literal, TypedDict


EnrichmentSetName = Literal["de_markers", "cell_marker", "sctype"]
# {enrichment set: {cell type: genes}}
ReferenceSets = dict[EnrichmentSetName, dict[str, set[str]]]
BiologicalComparisonRecord = dict[str, object]

# Whether a cluster's metrics were tested against random gene sets:
# "tested", "too_few_markers" (fewer than MIN_MARKER_GENES) or "sampling_failed" (no KDE could be fitted).
NullStatus = Literal["tested", "too_few_markers", "sampling_failed"]


class BiologicalMetricsRecord(TypedDict):
    """The fixed-shape record `compute_run_metrics` produces for one gene set vs one reference set.

    The gene set is a cluster's marker genes, or one of the random sets drawn for that cluster.
    `attach_tail_statistics` widens the cluster's records into the looser
    `BiologicalComparisonRecord` by merging in dynamically-named significance-stat
    fields (e.g. ``jaccard_z_score``, ``random_fpr_mean``).
    """

    enrichment_set: EnrichmentSetName
    cluster_id: str
    cluster_size: int
    n_marker_genes: int
    cell_type: str
    jaccard: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    specificity: float | None
    fpr: float | None
    intersection: int
    expected: float | None
    fold_enrichment: float | None


@dataclass(frozen=True)
class BiologicalEvaluation:
    """Result of evaluate_clustering_biologically for one run."""

    # One per (enrichment set, cluster, cell type), with the null statistics
    records: list[BiologicalComparisonRecord]
    # One value per column of the results table (bio_*)
    summary: dict[str, float | int | None]


class MembershipMetrics(TypedDict):
    tp: int
    fp: int
    fn: int
    tn: int
    jaccard: float
    precision: float
    recall: float
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
