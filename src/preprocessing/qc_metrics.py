import logging

from typing import List
import pandas as pd

from ..types import Species
from .gene_sets import apoptosis_genes, ensembl_id, is_ensembl_id, matching_genes, mito_genes, rrna_genes


logger = logging.getLogger(__name__)


def gene_set_fraction(data: pd.DataFrame, gene_list: List[str], name: str = "gene set") -> pd.Series:
    """
    Returns, for each cell, the fraction of its total counts coming from genes in gene_list.
    Genes are matched case-insensitively; cells get 0 when no gene in the list is present.
    name labels the gene set in the warning logged when none of its genes are found.
    """
    valid_genes = matching_genes(data, gene_list)

    if len(valid_genes) == 0 and len(gene_list) > 0:
        logger.warning(
            "None of the %s %s genes (matched by name or Ensembl ID) were found in the data columns, "
            "so every cell's %s fraction is 0",
            gene_set_size(gene_list), name, name)
        return pd.Series(0.0, index=data.index)

    subset_counts = data[valid_genes].sum(axis=1)
    total_counts = data.sum(axis=1)

    # Avoid division by zero by replacing 0 total counts with 1 (these cells will be dropped anyway or have 0 fraction)
    return subset_counts / total_counts.replace(0, 1)


def gene_set_size(gene_list: List[str]) -> int:
    """
    Returns the number of genes in gene_list. Annotation gene sets list every gene by both name and
    Ensembl ID, so their genes are counted by distinct Ensembl ID; lists without IDs count every entry.
    """
    gene_ids = {ensembl_id(gene) for gene in gene_list if is_ensembl_id(gene)}
    return len(gene_ids) if gene_ids else len(gene_list)


def cell_qc_metrics(data: pd.DataFrame, species: Species = "human") -> pd.DataFrame:
    """
    Returns one row per cell with the metrics the cell QC filters look at,
    sorted by count depth from high to low.

    Columns: count_depth (total counts), n_genes (genes with at least one count),
    and the fraction of counts from mitochondrial, rRNA and apoptosis genes.

    Example
    -------
    >>> metrics = cell_qc_metrics(raw_data, species="mouse")
    >>> metrics.head()   # deepest cells
    >>> metrics.tail()   # shallowest cells
    """
    metrics = pd.DataFrame({
        "count_depth": data.sum(axis=1),
        "n_genes": (data > 0).sum(axis=1),
        "mito_fraction": gene_set_fraction(data, mito_genes(species), "mitochondrial"),
        "rrna_fraction": gene_set_fraction(data, rrna_genes(species), "rRNA"),
        "apoptosis_fraction": gene_set_fraction(data, apoptosis_genes(species), "apoptosis"),
    })

    return metrics.sort_values("count_depth", ascending=False)
