from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .constants import DEFAULT_N_SAMPLES, DEFAULT_SEED, MIN_MARKER_GENES

EnrichmentSetName = Literal["de_markers", "cell_marker", "sctype"]
# {enrichment set: {cell type: genes}}
ReferenceSets = dict[EnrichmentSetName, dict[str, set[str]]]

# Whether a cluster was tested against random gene sets: "tested", "too_few_markers"
# (fewer than min_marker_genes) or "sampling_failed" (no KDE could be fitted to its markers).
NullStatus = Literal["tested", "too_few_markers", "sampling_failed"]


@dataclass(frozen=True)
class NullSettings:
    """How each cluster's null distribution is drawn."""

    n_samples: int = DEFAULT_N_SAMPLES  # random gene sets per cluster
    seed: int = DEFAULT_SEED
    min_marker_genes: int = MIN_MARKER_GENES  # fewer markers: scored, not tested


@dataclass(frozen=True)
class BiologicalEvaluation:
    """Result of evaluate_clustering_biologically for one run."""

    # One row per (enrichment set, cell type, cluster): metrics and null statistics
    records: pd.DataFrame
    # Marker genes of each cluster that has any
    cluster_markers: dict[str, list[str]]
    # One value per bio_* column of the results table
    summary: dict[str, float | int]
