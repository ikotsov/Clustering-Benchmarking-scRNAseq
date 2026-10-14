import re
from typing import Iterable, List, Optional, Set

import pandas as pd

from ..types import Species
from .gene_annotation import MITO_STATUS_COLUMN, load_gene_annotation
from .genes import HUMAN_APOPTOSIS_GENES, HUMAN_RRNA_GENES, MOUSE_APOPTOSIS_GENES, MOUSE_RRNA_GENES


def mito_genes(species: Species = "human", annotation: Optional[pd.DataFrame] = None) -> List[str]:
    # Names ("MT-CO1" / "mt-Co1") and Ensembl IDs of all mitochondrially encoded genes.
    return _flagged_genes(MITO_STATUS_COLUMN, species, annotation)


def apoptosis_genes(species: Species = "human", annotation: Optional[pd.DataFrame] = None) -> List[str]:
    gene_names = HUMAN_APOPTOSIS_GENES if species == "human" else MOUSE_APOPTOSIS_GENES
    return _with_ensembl_ids(gene_names, species, annotation)


def rrna_genes(species: Species = "human", annotation: Optional[pd.DataFrame] = None) -> List[str]:
    gene_names = HUMAN_RRNA_GENES if species == "human" else MOUSE_RRNA_GENES
    return _with_ensembl_ids(gene_names, species, annotation)


def matching_genes(data: pd.DataFrame, gene_list: List[str]) -> List[str]:
    """
    Returns the columns of data that appear in gene_list. Gene lists may mix gene names and Ensembl IDs,
    so they match datasets keyed by either.
    """
    gene_ids = {ensembl_id(gene) for gene in gene_list if is_ensembl_id(gene)}
    gene_names = {gene.upper()
                  for gene in gene_list if not is_ensembl_id(gene)}
    return [col for col in data.columns if is_in_gene_set(col, gene_ids, gene_names)]


def is_in_gene_set(gene: str, gene_ids: Set[str], gene_names: Set[str]) -> bool:
    """
    Looks a data column up by Ensembl ID if it is one (ignoring its version suffix),
    otherwise by gene name (case-insensitively).
    """
    if is_ensembl_id(gene):
        return ensembl_id(gene) in gene_ids
    return gene.upper() in gene_names


# Stable Ensembl gene ID with an optional version suffix, e.g. "ENSG00000198804.2" or "ENSMUSG00000064351".
ENSEMBL_ID_PATTERN = re.compile(r"^(ENS[A-Z]*G\d{11})(\.\d+)?$", re.IGNORECASE)


def is_ensembl_id(gene: str) -> bool:
    return ENSEMBL_ID_PATTERN.match(gene) is not None


def ensembl_id(gene: str) -> str:
    """
    Returns the uppercased Ensembl ID without its version suffix ("ENSG00000198804.2" -> "ENSG00000198804").
    """
    match = ENSEMBL_ID_PATTERN.match(gene)
    if match is None:
        raise ValueError(f"Not an Ensembl gene ID: {gene}")
    return match.group(1).upper()


def _flagged_genes(status_column: str, species: Species, annotation: Optional[pd.DataFrame]) -> List[str]:
    """
    Returns the gene names and Ensembl IDs of all genes flagged in a status column of the annotation.
    """
    if annotation is None:
        annotation = load_gene_annotation(species)
    flagged = annotation[~annotation[status_column].str.startswith("NO_")]
    return flagged["gene_name"].tolist() + flagged["gene_id"].tolist()


def _with_ensembl_ids(gene_names: Iterable[str], species: Species, annotation: Optional[pd.DataFrame]) -> List[str]:
    """
    Returns gene_names extended with their Ensembl IDs (names matched case-insensitively).
    """
    gene_names = list(gene_names)
    uppercase_names = {gene.upper() for gene in gene_names}
    if annotation is None:
        annotation = load_gene_annotation(species)
    is_listed = annotation["gene_name"].str.upper().isin(uppercase_names)
    return gene_names + annotation.loc[is_listed, "gene_id"].tolist()
