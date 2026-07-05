from collections.abc import Callable, Mapping

import pandas as pd

from .constants import DEFAULT_SEED, DEFAULT_STRATIFIED_SAMPLES
from .markers import (
    build_cluster_marker_enrichment_set,
    build_dataset_biological_enrichment_sets,
)
from .metrics import compute_biological_metrics_results
from .sampling import build_stratified_hvg_samples
from .types import BiologicalComparisonRecord, EnrichmentSetName
from src.types import Species


def evaluate_clustering_biologically(
    expression_data: pd.DataFrame,
    cell_type_labels: pd.Series[str],
    clustering_strategy: Callable[..., pd.Series],
    clustering_kwargs: Mapping[str, object] | None = None,
    enrichment_set_names: list[EnrichmentSetName] | None = None,
    species: Species = "human",
    n_samples: int = DEFAULT_STRATIFIED_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[BiologicalComparisonRecord]:
    """Run biological evaluation for one observed run and sampled-HVG reruns."""

    requested_enrichment_sets: list[EnrichmentSetName] = (
        enrichment_set_names if enrichment_set_names is not None else [
            "de_markers"]
    )
    # Keep deterministic order and avoid duplicate enrichment set requests.
    requested_enrichment_sets = list(dict.fromkeys(requested_enrichment_sets))
    reference_enrichment_sets = build_dataset_biological_enrichment_sets(
        expression_data=expression_data,
        cell_type_labels=cell_type_labels,
        enrichment_sets=requested_enrichment_sets,
        species=species,
    )

    gene_sampling_features = compute_gene_sampling_features(expression_data)
    hvg_genes = expression_data.columns.astype(str).tolist()
    sampled_hvg_sets = build_stratified_hvg_samples(
        gene_sampling_features=gene_sampling_features,
        hvg_genes=hvg_genes,
        n_samples=n_samples,
        seed=seed,
    )

    run_definitions: list[tuple[str, int, list[str]]] = [
        ("observed", 0, hvg_genes)]
    for sample_index, sampled_genes in enumerate(sampled_hvg_sets.control_samples, start=1):
        run_definitions.append(("sampled", sample_index, sampled_genes))

    all_genes = {str(gene) for gene in hvg_genes}
    effective_clustering_kwargs = clustering_kwargs if clustering_kwargs is not None else {}

    comparison_records: list[BiologicalComparisonRecord] = []

    for sample_type, sample_index, sampled_genes in run_definitions:
        sampled_expression_data = expression_data.loc[:, sampled_genes]
        clustering_labels = clustering_strategy(
            sampled_expression_data,
            **effective_clustering_kwargs,
        )
        predicted_labels = pd.Series(
            clustering_labels,
            index=sampled_expression_data.index,
            name="cluster",
        ).astype(str)

        cluster_marker_sets = build_cluster_marker_enrichment_set(
            expression_data=sampled_expression_data,
            predicted_cluster_labels=predicted_labels,
        )
        cluster_sizes = predicted_labels.value_counts()

        # for each cluster, compare its marker genes to the reference enrichment sets.
        for cluster_id, marker_genes in cluster_marker_sets.items():
            sorted_marker_genes = sorted(marker_genes)
            cluster_size = int(cluster_sizes.get(cluster_id, 0))

            if len(sorted_marker_genes) == 0:
                continue

            metrics = compute_biological_metrics_results(
                cluster_marker_genes=sorted_marker_genes,
                enrichment_sets=reference_enrichment_sets,
                all_genes=all_genes,
                cluster_id=cluster_id,
                cluster_size=cluster_size,
                sample_type=sample_type,
                sample_index=sample_index,
                sample_seed=seed + sample_index,
                run_hvg_count=len(sampled_genes),
            )
            comparison_records.extend(metrics)

    return comparison_records


def compute_gene_sampling_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build gene-level sampling features used to match random HVG controls.

    For each gene (column), this computes:
    - ``num_expressed_cells``: number of cells with expression > 0
    - ``mean_expression``: mean expression across all cells

    These features are later used by KDE-based sampling so random control gene
    sets are size-matched and sampled from genes with similar expression
    characteristics as the observed HVGs.
    """
    stats = pd.DataFrame(index=data.columns)
    # number of cells where the gene is expressed.
    stats["num_expressed_cells"] = (data > 0).sum(axis=0).astype(int)
    # mean expression of the gene across all cells.
    stats["mean_expression"] = data.mean(axis=0)
    stats.index.name = "gene"
    return stats
