import re
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from ..types import Species
from .gene_annotation import (
    ALIAS_COLUMN,
    ALIAS_SEPARATOR,
    APOPTOSIS_STATUS_COLUMN,
    MITO_STATUS_COLUMN,
    NO_ALIAS,
    RRNA_STATUS_COLUMN,
    load_gene_annotation,
)


def mito_genes(species: Species = "human", annotation: Optional[pd.DataFrame] = None) -> List[str]:
    # Names ("MT-CO1" / "mt-Co1") and Ensembl IDs of all mitochondrially encoded genes.
    return _flagged_genes(MITO_STATUS_COLUMN, species, annotation)


def apoptosis_genes(species: Species = "human", annotation: Optional[pd.DataFrame] = None) -> List[str]:
    return _flagged_genes(APOPTOSIS_STATUS_COLUMN, species, annotation)


def rrna_genes(species: Species = "human", annotation: Optional[pd.DataFrame] = None) -> List[str]:
    return _flagged_genes(RRNA_STATUS_COLUMN, species, annotation)


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


@dataclass(frozen=True)
class UnmatchedGenes:
    """
    Data columns that match no Ensembl ID or gene name of the annotation, split by what we could find for them.

    alias_matches: one row per (gene, annotation row whose aliases contain it), see alias_matches().
    unknown_ensembl_ids: Ensembl IDs missing from the annotation (e.g. retired in its Ensembl release).
    other: everything else, e.g. ERCC spike-ins or symbols that are not even an alias.
    """
    alias_matches: pd.DataFrame
    unknown_ensembl_ids: List[str]
    other: List[str]


def unmatched_genes(data: pd.DataFrame, annotation: pd.DataFrame) -> UnmatchedGenes:
    """
    Diagnoses the columns of data that the gene sets can never pick up, because they match no gene_id
    or gene_name of the annotation. Those still found through alias_names are reported with the annotation
    rows they point to, since an alias may belong to several genes.
    """
    annotated_genes = annotation["gene_name"].tolist() + annotation["gene_id"].tolist()
    matched = set(matching_genes(data, annotated_genes))
    unmatched = [col for col in data.columns if col not in matched]

    aliased = alias_matches(unmatched, annotation)
    aliased_genes = set(aliased["gene"])
    remaining = [gene for gene in unmatched if gene not in aliased_genes]
    return UnmatchedGenes(
        alias_matches=aliased,
        unknown_ensembl_ids=[gene for gene in remaining if is_ensembl_id(gene)],
        other=[gene for gene in remaining if not is_ensembl_id(gene)],
    )


def alias_matches(genes: List[str], annotation: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the annotation rows whose alias_names contain one of genes (case-insensitively).

    One row per gene and matching annotation row: a "gene" column, an "n_matches" column with the number of
    annotation rows that gene matches (more than 1 means the alias is ambiguous), then the annotation columns.
    Sorted with the most ambiguous genes first.
    """
    aliases = annotation.assign(alias=annotation[ALIAS_COLUMN].str.split(ALIAS_SEPARATOR)).explode("alias")
    aliases = aliases[aliases["alias"].notna() & (aliases["alias"] != NO_ALIAS)]
    aliases["alias"] = aliases["alias"].str.upper()

    lookup = pd.DataFrame({"gene": genes, "alias": [gene.upper() for gene in genes]})
    matches = (lookup.merge(aliases, on="alias")
               .drop(columns="alias")
               .drop_duplicates(subset=["gene", "gene_id"]))
    matches.insert(1, "n_matches", matches.groupby("gene")["gene_id"].transform("size"))
    return matches.sort_values(["n_matches", "gene"], ascending=[False, True]).reset_index(drop=True)


def duplicated_genes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnoses gene names that label more than one column of data. Empty if all gene names are unique.

    One row per repeated gene name, indexed by "gene": "n_columns" is the number of columns sharing it and
    "identical_counts" whether those columns hold the same values (a plain copy rather than distinct features
    that ended up with one name). Sorted with the most repeated genes first.
    """
    repeated = data.columns[data.columns.duplicated(keep=False)]
    rows = []
    for gene in repeated.unique():
        columns = data.iloc[:, np.flatnonzero(data.columns == gene)]
        identical = all(columns.iloc[:, 0].equals(columns.iloc[:, i]) for i in range(1, columns.shape[1]))
        rows.append({"gene": gene, "n_columns": columns.shape[1], "identical_counts": identical})
    report = pd.DataFrame(rows, columns=["gene", "n_columns", "identical_counts"]).set_index("gene")
    return report.sort_values("n_columns", ascending=False, kind="stable")


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
