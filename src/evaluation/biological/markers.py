import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Literal, TypeAlias, cast

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .constants import DE_MARKERS
from .types import EnrichmentSetName, ReferenceSets
from src.dataset_config import MarkerDatabaseConfig
from src.preprocessing.transforms import normalize_with_log_cpm
from src.types import PathLike, Species

logger = logging.getLogger(__name__)

# Configuration for DEG-based marker sets (as in the reference study).
DEFAULT_DEG_PVAL_CUTOFF = 0.01  # on the Benjamini-Hochberg adjusted p-value (scanpy's pval_cutoff)
DEFAULT_DEG_POSITIVE_LOGFC = 1.0
DEFAULT_DEG_NEGATIVE_LOGFC = -1.0
DEFAULT_DEG_METHOD = "wilcoxon"
# scanpy cannot test a group of one cell
MIN_GROUP_SIZE = 2

GeneMarkerGenes: TypeAlias = list[str]
GeneMarkerCellTypes: TypeAlias = dict[str, GeneMarkerGenes]
GeneMarkerTissues: TypeAlias = dict[str, GeneMarkerCellTypes]
GeneMarkerDatabases: TypeAlias = dict[str, GeneMarkerTissues]
GeneMarkerIndex: TypeAlias = dict[str, GeneMarkerDatabases]


def compute_log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """ln(CPM + 1) of raw counts (cells x genes), as float32 to halve the memory of all-gene matrices.

    The DE tests always run on log-CPM, whatever normalization the run clustered: marker genes
    describe the cells' biology, not the run's representation.
    """
    return normalize_with_log_cpm(counts).astype(np.float32)


def build_reference_sets(
    counts: pd.DataFrame,
    cell_type_labels: pd.Series,
    species: Species,
    annotation: pd.DataFrame | None = None,
    marker_databases: Mapping[str, MarkerDatabaseConfig] | None = None,
    marker_index: "GeneMarkerIndex | None" = None,
) -> ReferenceSets:
    """Build the reference sets of a dataset: {enrichment set: {cell type: genes}}.

    - "de_markers": DE markers of each ground-truth cell type (see build_de_marker_sets), on log-CPM of `counts`
    - each configured marker database: its cell types of the chosen tissues, mapped to the dataset's gene IDs

    Every set is restricted to the genes of `counts` (the universe); empty sets are dropped.
    """
    universe = counts.columns.astype(str).tolist()
    cells = counts.index.intersection(cell_type_labels.index)
    if len(cells) == 0:
        raise ValueError("No overlapping cells between counts and labels.")

    reference_sets: ReferenceSets = {
        DE_MARKERS: build_de_marker_sets(compute_log_cpm(counts.loc[cells]), cell_type_labels.loc[cells].astype(str)),
    }

    for database, settings in (marker_databases or {}).items():
        if marker_index is None:
            raise ValueError(f"Marker database '{database}' is configured, but no gene marker index was given")
        reference_sets[cast(EnrichmentSetName, database)] = build_database_marker_sets(
            database=database,
            species=species,
            tissues=settings["tissues"],
            cell_types=settings.get("cell_types"),
            marker_index=marker_index,
            universe=universe,
            annotation=annotation,
        )

    for name, sets in reference_sets.items():
        empty = sorted(cell_type for cell_type, genes in sets.items() if not genes)
        if empty:
            logger.warning("%s: no marker genes for %s; left out", name, ", ".join(empty))
        reference_sets[name] = {cell_type: genes for cell_type, genes in sets.items() if genes}

    return reference_sets


def build_de_marker_sets(
    expression_data: pd.DataFrame,
    group_labels: pd.Series,
) -> dict[str, set[str]]:
    """DE marker genes of each group (cell type or cluster) vs the rest of the cells.

    Wilcoxon test (`sc.tl.rank_genes_groups`), adjusted p < DEFAULT_DEG_PVAL_CUTOFF, then
    positive markers (logFC >= 1) plus negative markers (logFC <= -1) that are not a positive
    marker of any group. Groups with fewer than MIN_GROUP_SIZE cells get an empty set.

    expression_data must be non-negative log-normalized values (see compute_log_cpm).
    """
    labels = group_labels.loc[expression_data.index].astype(str)
    group_sizes = labels.value_counts()
    testable = sorted(group_sizes.index[group_sizes >= MIN_GROUP_SIZE])
    markers: dict[str, set[str]] = {str(group): set() for group in group_sizes.index}
    if len(testable) < 2:
        return markers

    adata = ad.AnnData(expression_data)
    adata.obs["group"] = pd.Categorical(labels.to_numpy())
    sc.tl.rank_genes_groups(adata, groupby="group", groups=testable, reference="rest", method=DEFAULT_DEG_METHOD)
    de_results = sc.get.rank_genes_groups_df(adata, group=testable, pval_cutoff=DEFAULT_DEG_PVAL_CUTOFF)
    if de_results.empty:
        return markers

    if "group" not in de_results.columns:  # scanpy leaves it out when a single group was tested
        de_results = de_results.assign(group=testable[0])
    de_results = de_results.assign(group_str=de_results["group"].astype(str))

    positive = _extract_deg_markers_by_group(de_results, direction="positive")
    negative = _extract_deg_markers_by_group(de_results, direction="negative")
    all_positive = set().union(*positive.values()) if positive else set()

    for group in markers:
        markers[group] = positive.get(group, set()) | (negative.get(group, set()) - all_positive)
    return markers


def build_database_marker_sets(
    database: str,
    species: Species,
    tissues: list[str],
    cell_types: list[str] | None,
    marker_index: GeneMarkerIndex,
    universe: list[str],
    annotation: pd.DataFrame | None = None,
) -> dict[str, set[str]]:
    """Marker genes of a database's cell types, from the given tissues, as gene IDs of the universe.

    A cell type found in several tissues gets the union of its markers. The databases list gene
    symbols; they are matched case-insensitively to the universe's gene IDs directly, and through
    the annotation's gene_id -> gene_name for datasets with Ensembl IDs. Symbols that match no
    gene of the universe are dropped.
    """
    bucket = _database_bucket(marker_index, database, species)
    unknown_tissues = sorted(set(tissues) - set(bucket))
    if unknown_tissues:
        raise ValueError(
            f"{database}: unknown tissues {unknown_tissues}; available: {sorted(bucket)}")

    symbols_by_cell_type: dict[str, set[str]] = defaultdict(set)
    for tissue in tissues:
        for cell_type, genes in bucket[tissue].items():
            if cell_types is None or cell_type in cell_types:
                symbols_by_cell_type[cell_type].update(genes)

    if cell_types is not None:
        missing = sorted(set(cell_types) - set(symbols_by_cell_type))
        if missing:
            raise ValueError(f"{database}: cell types {missing} are not in tissues {tissues}")

    genes_by_symbol = _genes_by_symbol(universe, annotation)
    return {
        cell_type: {gene for symbol in symbols for gene in genes_by_symbol.get(symbol.upper(), ())}
        for cell_type, symbols in sorted(symbols_by_cell_type.items())
    }


def load_gene_marker_index(path: PathLike) -> GeneMarkerIndex:
    """Load the gene marker index built by scripts/generate_gene_marker_index.py."""
    with open(path, "r", encoding="utf-8") as handle:
        return cast(GeneMarkerIndex, json.load(handle))


def _database_bucket(marker_index: GeneMarkerIndex, database: str, species: Species) -> GeneMarkerTissues:
    # CellMarker is species-specific; ScType is stored under "all" (see generate_gene_marker_index.py)
    species_key = "all" if database == "sctype" else species
    return marker_index.get(species_key, {}).get(database, {})


def _genes_by_symbol(universe: list[str], annotation: pd.DataFrame | None) -> dict[str, set[str]]:
    """Upper-case gene symbol -> genes of the universe it names."""
    names: dict[str, str] = {}
    if annotation is not None:
        names = annotation.dropna(subset=["gene_name"]).set_index("gene_id")["gene_name"].to_dict()

    genes_by_symbol: dict[str, set[str]] = defaultdict(set)
    for gene in universe:
        genes_by_symbol[gene.upper()].add(gene)
        name = names.get(gene)
        if isinstance(name, str) and name != "NO_NAME":
            genes_by_symbol[name.upper()].add(gene)
    return genes_by_symbol


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
