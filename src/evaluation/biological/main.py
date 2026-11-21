import logging

import numpy as np
import pandas as pd

from .markers import build_de_marker_sets, compute_log_cpm
from .metrics import (
    METRIC_NAMES,
    NULL_STATISTIC_NAMES,
    count_overlaps,
    null_statistics,
    overlap_metrics,
    summarize_biological_evaluation,
    untested_statistics,
)
from .sampling import MatchedSamplingError, build_matched_gene_samples, compute_gene_sampling_features
from .types import BiologicalEvaluation, NullSettings, NullStatus, ReferenceSets

logger = logging.getLogger(__name__)

RECORD_COLUMNS = ["enrichment_set", "cell_type", "cluster_id", "cluster_size", "n_marker_genes", "null_status",
                  *METRIC_NAMES, *NULL_STATISTIC_NAMES]


def evaluate_clustering_biologically(
    counts: pd.DataFrame,
    predicted_labels: pd.Series,
    reference_sets: ReferenceSets,
    null_settings: NullSettings = NullSettings(),
) -> BiologicalEvaluation:
    """Test whether each predicted cluster's marker genes match a cell type's marker genes beyond chance.

    1. Cluster markers: DE marker genes of each cluster vs the rest (log-CPM of `counts`).
    2. Observed metrics: each cluster's markers vs each reference set (Jaccard, precision, ...).
    3. Null, per cluster: random gene sets the size of its marker set, drawn from the universe and
       expression-matched to its markers, and their overlaps with the same reference sets. The
       clustering itself is never changed. Clusters with too few markers, or whose markers no KDE
       can be fitted to, are scored but not tested (null_status says why).
    4. The run is summarized per reference set by each cell type's best-matching cluster
       (see summarize_biological_evaluation).

    Parameters
    ----------
    counts : pd.DataFrame
        Filtered raw counts, cells x genes, of every cell in `predicted_labels`. Its genes are the
        universe: the DE tests run on them and the random sets are drawn from them.
    predicted_labels : pd.Series
        Cluster of each cell, from the clustering run being evaluated.
    reference_sets : ReferenceSets
        {enrichment set: {cell type: genes}}, within the universe (see build_reference_sets).
    """
    predicted_labels = predicted_labels.astype(str)
    counts = _cells_of_run(counts, predicted_labels)
    references = _flatten_reference_sets(reference_sets)
    reference_genes = references["genes"].tolist()

    gene_sampling_features = compute_gene_sampling_features(counts)
    cluster_markers = {
        cluster_id: sorted(genes)
        for cluster_id, genes in sorted(build_de_marker_sets(compute_log_cpm(counts), predicted_labels).items())
        if genes
    }
    cluster_sizes = predicted_labels.value_counts()

    tables = []
    for cluster_id, markers in cluster_markers.items():
        observed = count_overlaps(markers, reference_genes)
        status, random_overlaps = _draw_null_overlaps(markers, gene_sampling_features, reference_genes, null_settings)
        tables.append(references[["enrichment_set", "cell_type"]].assign(
            cluster_id=cluster_id,
            cluster_size=int(cluster_sizes[cluster_id]),
            n_marker_genes=len(markers),
            null_status=status,
            **overlap_metrics(observed, len(markers), references["n_genes"].to_numpy(), counts.shape[1]),
            **(null_statistics(observed, random_overlaps) if random_overlaps is not None else untested_statistics()),
        ))

    # No table when no cluster has markers (e.g. a single cluster): the run then scores 0
    records = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=RECORD_COLUMNS)
    summary = summarize_biological_evaluation(records, reference_sets, n_clusters=len(cluster_sizes))
    return BiologicalEvaluation(records=records, cluster_markers=cluster_markers, summary=summary)


def _cells_of_run(counts: pd.DataFrame, predicted_labels: pd.Series) -> pd.DataFrame:
    missing_cells = predicted_labels.index.difference(counts.index)
    if len(missing_cells) > 0:
        raise ValueError(
            f"{len(missing_cells)} cells of the run are not in the filtered counts (e.g. {missing_cells[0]}): "
            "the run must come from the same cell QC as the counts"
        )
    return counts.loc[predicted_labels.index]


def _flatten_reference_sets(reference_sets: ReferenceSets) -> pd.DataFrame:
    """One row per (enrichment set, cell type), with its genes and their number."""
    rows = [
        {"enrichment_set": set_name, "cell_type": cell_type, "genes": set(genes), "n_genes": len(genes)}
        for set_name, sets in reference_sets.items()
        for cell_type, genes in sets.items()
    ]
    return pd.DataFrame(rows, columns=["enrichment_set", "cell_type", "genes", "n_genes"])


def _draw_null_overlaps(
    markers: list[str],
    gene_sampling_features: pd.DataFrame,
    reference_genes: list[set[str]],
    settings: NullSettings,
) -> tuple[NullStatus, np.ndarray | None]:
    """Overlaps of expression-matched random sets with each reference set (random sets x reference sets),
    or None when the cluster is not tested."""
    if len(markers) < settings.min_marker_genes:
        return "too_few_markers", None
    try:
        random_sets = build_matched_gene_samples(gene_sampling_features, markers, settings.n_samples, settings.seed)
    except MatchedSamplingError as error:
        logger.warning("No random sets for a cluster of %s markers (%s); it is not tested", len(markers), error)
        return "sampling_failed", None
    return "tested", np.array([count_overlaps(random_set, reference_genes) for random_set in random_sets])
