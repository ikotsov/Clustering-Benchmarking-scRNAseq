import anndata as ad
from pathlib import Path
import pandas as pd
import scanpy as sc
from typing import Literal

from .types import EnrichmentSetName
from src.types import Species

# Configuration for DEG-based enrichment sets.
DEFAULT_DEG_PVAL_CUTOFF = 0.01
DEFAULT_DEG_POSITIVE_LOGFC = 1.0
DEFAULT_DEG_NEGATIVE_LOGFC = -1.0
DEFAULT_DEG_METHOD = "wilcoxon"
CELL_MARKER_FILENAME_BY_SPECIES: dict[Species, str] = {
    "human": "Cell_marker_Human.xlsx",
    "mouse": "Cell_marker_Mouse.xlsx",
}
SCTYPE_FILENAME = "ScTypeDB_full.xlsx"
MARKER_DATABASE_DIR = Path(__file__).resolve(
).parents[3] / "data" / "gene_markers"


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
        return build_cell_marker_enrichment_set(species=species)
    if enrichment_set == "sctype":
        return build_sctype_enrichment_set()

    raise ValueError(f"Unsupported enrichment set: {enrichment_set}")


def build_cell_marker_enrichment_set(
    species: Species,
) -> dict[str, set[str]]:
    """Build marker sets grouped by CellMarker cell_name for a species."""
    file_name = CELL_MARKER_FILENAME_BY_SPECIES[species]
    marker_path = MARKER_DATABASE_DIR / file_name
    if not marker_path.exists():
        raise FileNotFoundError(
            f"Marker database file not found: {marker_path}")

    marker_data = pd.read_excel(marker_path)

    selected = marker_data.loc[:, ["cell_name", "marker"]].dropna()
    selected["cell_name"] = selected["cell_name"].astype(str).str.strip()
    selected["marker"] = selected["marker"].astype(str).str.strip()
    selected = selected[
        (selected["cell_name"] != "")
        & (selected["marker"] != "")
    ]

    return {
        str(cell_name): {
            str(gene)
            for gene in genes.tolist()
            if pd.notna(gene) and str(gene).strip() != ""
        }
        for cell_name, genes in selected.groupby("cell_name")["marker"]
    }


def build_sctype_enrichment_set() -> dict[str, set[str]]:
    """Build ScType marker sets grouped by cellName across all tissues."""
    marker_path = MARKER_DATABASE_DIR / SCTYPE_FILENAME
    if not marker_path.exists():
        raise FileNotFoundError(
            f"Marker database file not found: {marker_path}")

    sctype_data = pd.read_excel(marker_path)

    selected = sctype_data.loc[:, ["cellName",
                                   "geneSymbolmore1", "geneSymbolmore2"]].copy()
    selected["cellName"] = selected["cellName"].astype(str).str.strip()
    selected = selected[selected["cellName"] != ""]
    selected["genes"] = selected.apply(_parse_gene_symbols, axis=1)

    grouped = selected.groupby("cellName")["genes"].agg(
        lambda gene_sets: set().union(*gene_sets)
    )
    return {str(cell_name): set(genes) for cell_name, genes in grouped.items()}


def _parse_gene_symbols(row: pd.Series) -> set[str]:
    """Convert ScType geneSymbolmore columns into one deduplicated gene set."""
    genes: set[str] = set()
    for column_name in ("geneSymbolmore1", "geneSymbolmore2"):
        genes.update(_parse_gene_symbol_value(row[column_name]))

    return genes


def _parse_gene_symbol_value(value: str) -> set[str]:
    """Split a comma-separated gene list into a set of symbols."""
    value_str = value.strip()
    if value_str == "" or value_str.lower() == "nan":
        return set()

    return {
        token.strip()
        for token in value_str.split(",")
        if token.strip() != ""
    }


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
