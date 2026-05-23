import json
import anndata as ad
from pathlib import Path
import pandas as pd
import scanpy as sc
from typing import Literal, TypeAlias, cast

from .types import EnrichmentSetName
from src.types import Species

# Configuration for DEG-based enrichment sets.
DEFAULT_DEG_PVAL_CUTOFF = 0.01
DEFAULT_DEG_POSITIVE_LOGFC = 1.0
DEFAULT_DEG_NEGATIVE_LOGFC = -1.0
DEFAULT_DEG_METHOD = "wilcoxon"

MARKER_DATABASE_DIR = Path(__file__).resolve(
).parents[3] / "data" / "gene_markers"
GENE_MARKER_INDEX_FILENAME = "gene_marker_index.json"
GENE_MARKER_INDEX_PATH = MARKER_DATABASE_DIR / \
    "outputs" / GENE_MARKER_INDEX_FILENAME
GeneMarkerGenes: TypeAlias = list[str]
GeneMarkerCellTypes: TypeAlias = dict[str, GeneMarkerGenes]
GeneMarkerTissues: TypeAlias = dict[str, GeneMarkerCellTypes]
GeneMarkerDatabases: TypeAlias = dict[str, GeneMarkerTissues]
GeneMarkerIndex: TypeAlias = dict[str, GeneMarkerDatabases]
GeneMarkerCollapsedCellTypes: TypeAlias = dict[str, set[str]]


def build_dataset_biological_enrichment_sets(
    expression_data: pd.DataFrame,
    cell_type_labels: pd.Series[str],
    enrichment_sets: list[EnrichmentSetName],
    species: Species = "human",
) -> dict[EnrichmentSetName, dict[str, set[str]]]:
    """Build all requested biological enrichment sets for one dataset."""
    return {
        enrichment_set: build_biological_enrichment_set(
            expression_data,
            cell_type_labels,
            enrichment_set,
            species=species,
        )
        for enrichment_set in enrichment_sets
    }


def build_biological_enrichment_set(
    expression_data: pd.DataFrame,
    cell_type_labels: pd.Series[str],
    enrichment_set: EnrichmentSetName,
    species: Species = "human",
) -> dict[str, set[str]]:
    """Build one supported enrichment set for a dataset."""
    if enrichment_set == "de_markers":
        return build_de_marker_enrichment_set(expression_data, cell_type_labels)
    if enrichment_set == "cell_marker":
        return get_cell_marker_enrichment_set(species=species)
    if enrichment_set == "sctype":
        return get_sctype_enrichment_set()

    raise ValueError(f"Unsupported enrichment set: {enrichment_set}")


def build_de_marker_enrichment_set(
    expression_data: pd.DataFrame,
    cell_type_labels: pd.Series[str],
) -> dict[str, set[str]]:
    """Build DEG-based enrichment sets per cell type.

    Positive markers (logFC >= threshold) are combined with negative markers
    (logFC <= threshold) after removing genes that are positive in any cell type.
    """
    common_cells = expression_data.index.intersection(cell_type_labels.index)
    if len(common_cells) == 0:
        raise ValueError(
            "No overlapping cells between expression data and labels.")

    matrix = _prepare_rank_genes_groups_matrix(
        expression_data.loc[common_cells])
    labels = cell_type_labels.loc[common_cells]

    adata = ad.AnnData(matrix)
    adata.obs["cell_type"] = labels

    sc.tl.rank_genes_groups(
        adata,
        groupby="cell_type",
        method=DEFAULT_DEG_METHOD,
    )
    de_results = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        pval_cutoff=DEFAULT_DEG_PVAL_CUTOFF,
    )

    if de_results.empty:
        return {str(cell_type): set() for cell_type in labels.unique()}

    de_results = de_results.copy()
    de_results["group_str"] = de_results["group"].astype(str)
    cell_types = de_results["group_str"].unique()

    positive_markers_by_cell_type = _extract_deg_markers_by_group(
        de_results,
        direction="positive",
    )
    negative_markers_by_cell_type = _extract_deg_markers_by_group(
        de_results,
        direction="negative",
    )

    all_positive = set().union(*positive_markers_by_cell_type.values()
                               ) if positive_markers_by_cell_type else set()

    cleaned_negative_markers_by_cell_type = {
        cell_type: genes - all_positive
        for cell_type, genes in negative_markers_by_cell_type.items()
    }

    return {
        cell_type: cleaned_negative_markers_by_cell_type.get(cell_type, set())
        | positive_markers_by_cell_type.get(cell_type, set())
        for cell_type in cell_types
    }


def get_cell_marker_enrichment_set(
    species: Species,
) -> GeneMarkerCollapsedCellTypes:
    """Build marker sets grouped by CellMarker cell_name for a species."""
    marker_index = load_gene_marker_index()
    species_bucket = marker_index.get(species, {})
    cell_marker_bucket = species_bucket.get("cell_marker", {})
    return _collapse_marker_index(cell_marker_bucket)


def get_sctype_enrichment_set() -> GeneMarkerCollapsedCellTypes:
    """Build ScType marker sets grouped by cellName across all tissues."""
    marker_index = load_gene_marker_index()
    all_species_bucket = marker_index.get("all", {})
    sctype_bucket = all_species_bucket.get("sctype", {})
    return _collapse_marker_index(sctype_bucket)


def load_gene_marker_index() -> GeneMarkerIndex:
    """Load the prebuilt gene marker index from the outputs folder."""
    if not GENE_MARKER_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Gene marker index file not found: {GENE_MARKER_INDEX_PATH}"
        )

    with GENE_MARKER_INDEX_PATH.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    return cast(GeneMarkerIndex, loaded)


def _collapse_marker_index(
    tissue_bucket: GeneMarkerTissues,
) -> GeneMarkerCollapsedCellTypes:
    collapsed: GeneMarkerCollapsedCellTypes = {}
    for cell_types in tissue_bucket.values():
        for cell_name, genes in cell_types.items():
            collapsed.setdefault(cell_name, set()).update(genes)

    return collapsed


def build_cluster_marker_enrichment_set(
    expression_data: pd.DataFrame,
    predicted_cluster_labels: pd.Series[str],
) -> dict[str, set[str]]:
    """Build DEG-based marker sets per predicted cluster."""
    matrix = _prepare_rank_genes_groups_matrix(expression_data)
    labels = predicted_cluster_labels

    adata = ad.AnnData(matrix)
    adata.obs["cluster"] = labels

    sc.tl.rank_genes_groups(
        adata,
        groupby="cluster",
        method=DEFAULT_DEG_METHOD,
    )
    de_results = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        pval_cutoff=DEFAULT_DEG_PVAL_CUTOFF,
    )

    if de_results.empty:
        return {str(cluster): set() for cluster in labels.unique()}

    de_results = de_results.copy()
    de_results["group_str"] = de_results["group"].astype(str)
    cluster_ids = de_results["group_str"].unique()

    positive_markers_by_cluster = _extract_deg_markers_by_group(
        de_results,
        direction="positive",
    )
    negative_markers_by_cluster = _extract_deg_markers_by_group(
        de_results,
        direction="negative",
    )

    all_positive = (
        set().union(*positive_markers_by_cluster.values())
        if positive_markers_by_cluster
        else set()
    )

    cleaned_negative_markers_by_cluster = {
        cluster_id: genes - all_positive
        for cluster_id, genes in negative_markers_by_cluster.items()
    }

    return {
        cluster_id: cleaned_negative_markers_by_cluster.get(cluster_id, set())
        | positive_markers_by_cluster.get(cluster_id, set())
        for cluster_id in cluster_ids
    }


def _extract_deg_markers_by_group(
    de_results: pd.DataFrame,
    direction: Literal["positive", "negative"],
) -> dict[str, set[str]]:
    if direction == "positive":
        selected = de_results.loc[
            de_results["logfoldchanges"] >= DEFAULT_DEG_POSITIVE_LOGFC,
            ["group_str", "names"],
        ]
    else:
        selected = de_results.loc[
            de_results["logfoldchanges"] <= DEFAULT_DEG_NEGATIVE_LOGFC,
            ["group_str", "names"],
        ]

    selected = selected.dropna(subset=["names"])

    return {
        str(group_id): {str(gene) for gene in genes.tolist() if pd.notna(gene)}
        for group_id, genes in selected.groupby("group_str")["names"]
    }

# TODO: Remove the below, with pearson's we will use a t-test instead of a wilcoxon text.


def _prepare_rank_genes_groups_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Ensure rank_genes_groups receives non-negative expression values.

    Pearson residual-based inputs can contain negatives, which produce invalid
    log2 fold changes inside Scanpy's rank_genes_groups implementation.
    """
    return matrix.clip(lower=0)
