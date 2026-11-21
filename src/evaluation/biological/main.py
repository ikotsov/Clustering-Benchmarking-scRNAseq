import pandas as pd

from .constants import DEFAULT_SEED, DEFAULT_STRATIFIED_SAMPLES
from .markers import (
    build_cluster_marker_enrichment_set,
    build_dataset_biological_enrichment_sets,
)
from .metrics import attach_tail_statistics, compute_run_metrics
from .sampling import build_matched_gene_samples, compute_gene_sampling_features
from .types import BiologicalComparisonRecord, BiologicalMetricsRecord, EnrichmentSetName
from src.types import Species


def evaluate_clustering_biologically(
    expression_data: pd.DataFrame,
    counts: pd.DataFrame,
    cell_type_labels: pd.Series,
    predicted_labels: pd.Series,
    enrichment_set_names: list[EnrichmentSetName] | None = None,
    species: Species = "human",
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[BiologicalComparisonRecord]:
    """Test whether each predicted cluster's marker genes match a cell type's marker genes beyond chance.

    1. Reference sets: marker genes of each ground-truth cell type (`enrichment_set_names`).
    2. Cluster markers: marker genes of each predicted cluster (DE test, cluster vs rest).
    3. Observed metrics: each cluster's markers vs each reference set (Jaccard, precision, ...).
    4. Null, per cluster: `n_samples` random gene sets the size of the cluster's marker set,
       expression-matched to it (KDE on raw-count features), each scored against the same
       reference sets. The clustering itself is never changed.
    5. Each observed metric gets a z-score, an empirical p-value and the null's 2.5-97.5% interval.

    Parameters
    ----------
    expression_data : pd.DataFrame
        Cells x genes, normalized (the no-PCA matrix). Its genes are the universe: the DE tests run on
        them, and the random sets are drawn from them.
    counts : pd.DataFrame
        Raw counts, cells x genes; must contain every gene and cell of `expression_data`. Only used to
        describe the genes' expression level for the matching.
    cell_type_labels : pd.Series
        Ground-truth cell type of each cell.
    predicted_labels : pd.Series
        Cluster of each cell, from the clustering run being evaluated.

    Returns
    -------
    One record per (enrichment set, cluster, cell type), with the observed metrics and their null statistics.
    Clusters without marker genes have no records.
    """
    requested_enrichment_sets: list[EnrichmentSetName] = list(
        dict.fromkeys(enrichment_set_names or ["de_markers"]))
    reference_enrichment_sets = build_dataset_biological_enrichment_sets(
        expression_data=expression_data,
        cell_type_labels=cell_type_labels,
        enrichment_sets=requested_enrichment_sets,
        species=species,
    )

    universe = expression_data.columns.astype(str).tolist()
    all_genes = set(universe)
    gene_sampling_features = compute_gene_sampling_features(
        counts.loc[expression_data.index, universe])

    predicted_labels = predicted_labels.loc[expression_data.index].astype(str).rename("cluster")
    cluster_marker_sets = build_cluster_marker_enrichment_set(
        expression_data=expression_data,
        predicted_cluster_labels=predicted_labels,
    )
    cluster_sizes = predicted_labels.value_counts()

    observed_records: list[BiologicalMetricsRecord] = []
    sampled_records: list[BiologicalMetricsRecord] = []

    for cluster_id in sorted(cluster_marker_sets):
        marker_genes = sorted(cluster_marker_sets[cluster_id])
        if len(marker_genes) == 0:
            continue

        def score(genes: list[str], sample_type: str, sample_index: int) -> list[BiologicalMetricsRecord]:
            return compute_run_metrics(
                cluster_marker_genes=genes,
                enrichment_sets=reference_enrichment_sets,
                all_genes=all_genes,
                cluster_id=cluster_id,
                cluster_size=int(cluster_sizes.get(cluster_id, 0)),
                sample_type=sample_type,
                sample_index=sample_index,
                sample_seed=seed,
                run_hvg_count=len(universe),
            )

        observed_records.extend(score(marker_genes, "observed", 0))

        random_marker_sets = build_matched_gene_samples(
            gene_sampling_features=gene_sampling_features,
            observed_genes=marker_genes,
            n_samples=n_samples,
            seed=seed,
        )
        for sample_index, random_genes in enumerate(random_marker_sets, start=1):
            sampled_records.extend(score(random_genes, "sampled", sample_index))

    # The random sets are only needed for their statistics, which attach_tail_statistics
    # writes onto the observed records (random_<metric>_mean/std/n, <metric>_ci_lower/upper).
    return attach_tail_statistics(observed_records, sampled_records)
