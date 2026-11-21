"""Marker gene sets: DE markers of groups of cells (cell types or clusters) and marker databases."""
import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import cast

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

# DE marker selection, as in the reference study
DEFAULT_DEG_PVAL_CUTOFF = 0.01  # on the Benjamini-Hochberg adjusted p-value (scanpy's pval_cutoff)
DEFAULT_DEG_POSITIVE_LOGFC = 1.0
DEFAULT_DEG_NEGATIVE_LOGFC = -1.0
DEFAULT_DEG_METHOD = "wilcoxon"
# scanpy cannot test a group of one cell
MIN_GROUP_SIZE = 2

# Gene marker index (scripts/generate_gene_marker_index.py): species -> database -> tissue -> cell type -> genes
TissueMarkers = dict[str, dict[str, list[str]]]
GeneMarkerIndex = dict[str, dict[str, TissueMarkers]]


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
    marker_index: GeneMarkerIndex | None = None,
) -> ReferenceSets:
    """Build the reference sets of a dataset: {enrichment set: {cell type: genes}}.

    - "de_markers": DE markers of each ground-truth cell type (see build_de_marker_sets), on log-CPM of `counts`
    - each configured marker database: its cell types of the chosen tissues, as gene IDs of `counts`

    Every set is within the genes of `counts` (the universe); empty sets are left out.
    """
    cells = counts.index.intersection(cell_type_labels.index)
    if len(cells) == 0:
        raise ValueError("No overlapping cells between counts and labels.")

    reference_sets: ReferenceSets = {
        DE_MARKERS: build_de_marker_sets(compute_log_cpm(counts.loc[cells]), cell_type_labels.loc[cells]),
    }
    if marker_databases:
        if marker_index is None:
            raise ValueError("Marker databases are configured, but no gene marker index was given")
        genes_by_symbol = index_genes_by_symbol(counts.columns.astype(str).tolist(), annotation)
        for database, settings in marker_databases.items():
            reference_sets[cast(EnrichmentSetName, database)] = build_database_marker_sets(
                database, settings, species, marker_index, genes_by_symbol)

    return _drop_empty_sets(reference_sets)


def build_de_marker_sets(expression_data: pd.DataFrame, group_labels: pd.Series) -> dict[str, set[str]]:
    """DE marker genes of each group (cell type or cluster) vs the rest of the cells.

    Wilcoxon test, adjusted p < DEFAULT_DEG_PVAL_CUTOFF, then positive markers (logFC >= 1) plus
    negative markers (logFC <= -1) that are not a positive marker of any group. Groups with fewer
    than MIN_GROUP_SIZE cells get an empty set.

    expression_data must be non-negative log-normalized values (see compute_log_cpm).
    """
    labels = group_labels.loc[expression_data.index].astype(str)
    group_sizes = labels.value_counts()
    markers: dict[str, set[str]] = {group: set() for group in group_sizes.index}
    testable = sorted(group_sizes.index[group_sizes >= MIN_GROUP_SIZE])
    if len(testable) < 2:
        return markers

    de_results = _rank_genes(expression_data, labels, testable)
    positive = _genes_by_group(de_results[de_results["logfoldchanges"] >= DEFAULT_DEG_POSITIVE_LOGFC])
    negative = _genes_by_group(de_results[de_results["logfoldchanges"] <= DEFAULT_DEG_NEGATIVE_LOGFC])
    all_positive = set().union(*positive.values())

    for group in markers:
        markers[group] = positive.get(group, set()) | (negative.get(group, set()) - all_positive)
    return markers


def build_database_marker_sets(
    database: str,
    settings: MarkerDatabaseConfig,
    species: Species,
    marker_index: GeneMarkerIndex,
    genes_by_symbol: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """Marker genes of a database's cell types from the configured tissues, as gene IDs of the universe.

    A cell type found in several tissues gets the union of its markers. Symbols that name no gene of
    the universe (see index_genes_by_symbol) are dropped.
    """
    try:
        symbols_by_cell_type = _select_database_symbols(_database_tissues(marker_index, database, species), settings)
    except ValueError as error:
        raise ValueError(f"{database}: {error}") from error
    return {
        cell_type: {gene for symbol in symbols for gene in genes_by_symbol.get(symbol.upper(), ())}
        for cell_type, symbols in sorted(symbols_by_cell_type.items())
    }


def index_genes_by_symbol(universe: list[str], annotation: pd.DataFrame | None) -> dict[str, set[str]]:
    """Upper-case gene symbol -> genes of the universe it names.

    The databases list gene symbols: they match a universe of symbols directly (case-insensitively,
    so human symbols also match mouse ones), and a universe of Ensembl IDs through the annotation's
    gene_id -> gene_name.
    """
    names: dict[str, str] = {}
    if annotation is not None:
        names = annotation.dropna(subset=["gene_name"]).set_index("gene_id")["gene_name"].to_dict()

    genes_by_symbol: dict[str, set[str]] = defaultdict(set)
    for gene in universe:
        genes_by_symbol[gene.upper()].add(gene)
        name = names.get(gene)
        if name and name != "NO_NAME":
            genes_by_symbol[name.upper()].add(gene)
    return genes_by_symbol


def load_gene_marker_index(path: PathLike) -> GeneMarkerIndex:
    """Load the gene marker index built by scripts/generate_gene_marker_index.py."""
    with open(path, "r", encoding="utf-8") as handle:
        return cast(GeneMarkerIndex, json.load(handle))


def _rank_genes(expression_data: pd.DataFrame, labels: pd.Series, groups: list[str]) -> pd.DataFrame:
    """Genes passing the adjusted p-value cutoff for each group vs the rest: group, names, logfoldchanges, ..."""
    adata = ad.AnnData(expression_data)
    adata.obs["group"] = pd.Categorical(labels.to_numpy())
    sc.tl.rank_genes_groups(adata, groupby="group", groups=groups, reference="rest", method=DEFAULT_DEG_METHOD)
    de_results = sc.get.rank_genes_groups_df(adata, group=groups, pval_cutoff=DEFAULT_DEG_PVAL_CUTOFF)
    if "group" not in de_results.columns:  # scanpy leaves it out when a single group was asked for
        de_results = de_results.assign(group=groups[0])
    return de_results.assign(group=de_results["group"].astype(str))


def _genes_by_group(de_results: pd.DataFrame) -> dict[str, set[str]]:
    return {str(group): set(genes.dropna().astype(str)) for group, genes in de_results.groupby("group")["names"]}


def _database_tissues(marker_index: GeneMarkerIndex, database: str, species: Species) -> TissueMarkers:
    # CellMarker is species-specific; ScType is stored under "all" (see generate_gene_marker_index.py)
    species_key = "all" if database == "sctype" else species
    return marker_index.get(species_key, {}).get(database, {})


def _select_database_symbols(tissues: TissueMarkers, settings: MarkerDatabaseConfig) -> dict[str, set[str]]:
    """Gene symbols of each cell type of the configured tissues (only the configured cell types, if any)."""
    unknown_tissues = sorted(set(settings["tissues"]) - set(tissues))
    if unknown_tissues:
        raise ValueError(f"Unknown tissues {unknown_tissues}; available: {sorted(tissues)}")

    wanted_cell_types = settings.get("cell_types")
    symbols_by_cell_type: dict[str, set[str]] = defaultdict(set)
    for tissue in settings["tissues"]:
        for cell_type, symbols in tissues[tissue].items():
            if wanted_cell_types is None or cell_type in wanted_cell_types:
                symbols_by_cell_type[cell_type].update(symbols)

    missing = sorted(set(wanted_cell_types or []) - set(symbols_by_cell_type))
    if missing:
        raise ValueError(f"Cell types {missing} are not in tissues {settings['tissues']}")
    return symbols_by_cell_type


def _drop_empty_sets(reference_sets: ReferenceSets) -> ReferenceSets:
    """Leave out cell types without genes, and reference sets without cell types, with a warning."""
    kept: ReferenceSets = {}
    for name, sets in reference_sets.items():
        empty = sorted(cell_type for cell_type, genes in sets.items() if not genes)
        if empty:
            logger.warning("%s: no marker genes for %s; left out", name, ", ".join(empty))
        non_empty = {cell_type: genes for cell_type, genes in sets.items() if genes}
        if non_empty:
            kept[name] = non_empty
        else:
            logger.warning("%s: no cell type has marker genes; the reference set is left out", name)
    return kept
