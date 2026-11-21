import logging

import pandas as pd

from .constants import DEFAULT_SEED, DEFAULT_STRATIFIED_SAMPLES, MIN_MARKER_GENES
from .markers import build_de_marker_sets, compute_log_cpm
from .metrics import attach_tail_statistics, compute_run_metrics, summarize_biological_evaluation
from .sampling import MatchedSamplingError, build_matched_gene_samples, compute_gene_sampling_features
from .types import BiologicalEvaluation, BiologicalMetricsRecord, NullStatus, ReferenceSets

logger = logging.getLogger(__name__)


def evaluate_clustering_biologically(
    counts: pd.DataFrame,
    predicted_labels: pd.Series,
    reference_sets: ReferenceSets,
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
    min_marker_genes: int = MIN_MARKER_GENES,
) -> BiologicalEvaluation:
    """Test whether each predicted cluster's marker genes match a cell type's marker genes beyond chance.

    1. Cluster markers: DE marker genes of each cluster vs the rest (log-CPM of `counts`).
    2. Observed metrics: each cluster's markers vs each reference set (Jaccard, precision, ...).
    3. Null, per cluster: `n_samples` random gene sets the size of its marker set, drawn from the
       universe and expression-matched to its markers (KDE on raw-count features), each scored
       against the same reference sets. The clustering itself is never changed.
    4. Each observed metric gets a z-score, an empirical p-value and the null's 2.5-97.5% interval.
       Clusters with fewer than `min_marker_genes` markers, or whose markers no KDE can be fitted to,
       are scored but not tested (null_status says why).
    5. The run is summarized per reference set by each cell type's best-matching cluster
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
    missing_cells = predicted_labels.index.difference(counts.index)
    if len(missing_cells) > 0:
        raise ValueError(
            f"{len(missing_cells)} cells of the run are not in the filtered counts (e.g. {missing_cells[0]}): "
            "the run must come from the same cell QC as the counts"
        )
    counts = counts.loc[predicted_labels.index]
    universe = set(counts.columns.astype(str))

    gene_sampling_features = compute_gene_sampling_features(counts)
    cluster_marker_sets = build_de_marker_sets(compute_log_cpm(counts), predicted_labels)
    cluster_sizes = predicted_labels.value_counts()

    observed_records: list[BiologicalMetricsRecord] = []
    sampled_records: list[BiologicalMetricsRecord] = []
    marker_genes_by_cluster: dict[str, list[str]] = {}
    null_status: dict[str, NullStatus] = {}

    for cluster_id in sorted(cluster_marker_sets):
        marker_genes = sorted(cluster_marker_sets[cluster_id])
        if not marker_genes:
            continue
        cluster_size = int(cluster_sizes[cluster_id])
        marker_genes_by_cluster[cluster_id] = marker_genes
        observed_records.extend(compute_run_metrics(
            marker_genes, reference_sets, universe, cluster_id, cluster_size))

        if len(marker_genes) < min_marker_genes:
            null_status[cluster_id] = "too_few_markers"
            continue
        try:
            random_sets = build_matched_gene_samples(gene_sampling_features, marker_genes, n_samples, seed)
        except MatchedSamplingError as error:
            logger.warning("Cluster %s: no random sets (%s); its metrics are not tested", cluster_id, error)
            null_status[cluster_id] = "sampling_failed"
            continue
        null_status[cluster_id] = "tested"
        for random_genes in random_sets:
            sampled_records.extend(compute_run_metrics(
                random_genes, reference_sets, universe, cluster_id, cluster_size))

    records = attach_tail_statistics(observed_records, sampled_records)
    for record in records:
        cluster_id = str(record["cluster_id"])
        record["null_status"] = null_status[cluster_id]
        record["marker_genes"] = marker_genes_by_cluster[cluster_id]

    summary = summarize_biological_evaluation(
        records,
        reference_sets,
        n_clusters=len(cluster_sizes),
        n_clusters_without_markers=len(cluster_sizes) - len(marker_genes_by_cluster),
        n_clusters_untested=sum(status != "tested" for status in null_status.values()),
    )
    return BiologicalEvaluation(records=records, summary=summary)
