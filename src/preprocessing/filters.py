from typing import List
import pandas as pd
import scanpy as sc

from .types import Species
from .genes import HUMAN_APOPTOSIS_GENES, HUMAN_RRNA_GENES, MOUSE_APOPTOSIS_GENES, MOUSE_RRNA_GENES
from ..constants import GENE_MAGNITUDE_THRESHOLD


def filter_low_magnitude_genes(data: pd.DataFrame, min_count: int = GENE_MAGNITUDE_THRESHOLD) -> pd.DataFrame:
    """
    Removes genes that never exceed a specific count threshold.
    (by default removes genes containing only 0s and 1s).

    Parameters
    ----------
    data : pd.DataFrame
    min_count : int, default=2
        A gene is kept only if at least one cell has a count >= min_count.

    Returns
    -------
    data_filtered : pd.DataFrame

    Example
    --------
    >>> data
              Gene_A  Gene_Binary  Gene_Zero
    Sample_1     10        1           0
    Sample_2      5        0           0
    Sample_3      2        1           0

    >>> filter_low_magnitude_genes(data, min_count=2)
    # Gene_Binary is removed (max value is 1)
    # Gene_Zero is removed (max value is 0)
              Gene_A
    Sample_1     10
    Sample_2      5
    Sample_3      2
    """
    # Check the max value of each column (gene)
    # If max value < 2, it implies the gene only has 0s and 1s.
    mask = data.max(axis=0) >= min_count
    data_filtered = data.loc[:, mask]

    dropped = data.shape[1] - data_filtered.shape[1]

    print(f"  • Dropped {dropped} low-magnitude genes")

    return data_filtered


def filter_high_mito_cells(data: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Removes cells with high mitochondrial expression (indicative of broken cells).
    """
    # By transforming gene names to uppercase, we can catch both "MT-" and "mt-" prefixes. "mt-" is common in mouse datasets.
    mt_genes = [gene for gene in data.columns if gene.upper().startswith("MT-")]

    return filter_cells_by_fraction(
        data,
        gene_list=mt_genes,
        threshold=threshold,
        filter_name="mitochondrial"
    )


def filter_high_apoptosis_cells(data: pd.DataFrame, threshold: float = 0.05, species: Species = "human") -> pd.DataFrame:
    """
    Removes cells with high expression of apoptosis-related genes (indicative of cell stress).
    """
    return filter_cells_by_fraction(
        data,
        gene_list=HUMAN_APOPTOSIS_GENES if species == "human" else MOUSE_APOPTOSIS_GENES,
        threshold=threshold,
        filter_name="apoptosis"
    )


def filter_high_rrna_cells(data: pd.DataFrame, threshold: float = 0.05, species: Species = "human") -> pd.DataFrame:
    """
    Removes cells with high rRNA expression (indicative of technical noise).
    """
    return filter_cells_by_fraction(
        data,
        gene_list=HUMAN_RRNA_GENES if species == "human" else MOUSE_RRNA_GENES,
        threshold=threshold,
        filter_name="rRNA"
    )


def filter_cells_by_fraction(data: pd.DataFrame, gene_list: List[str], threshold: float, filter_name: str = "generic") -> pd.DataFrame:
    """
    Removes cells with high expression of a specific gene set.
    """
    valid_genes = [gene for gene in gene_list if gene.upper() in data.columns]

    if len(valid_genes) == 0:
        return data

    subset_counts = data[valid_genes].sum(axis=1)
    total_counts = data.sum(axis=1)

    # Avoid division by zero by replacing 0 total counts with 1 (these cells will be dropped anyway or have 0 fraction)
    expression_ratio = subset_counts / total_counts.replace(0, 1)

    # Keep cells where ratio is less than or equal to the threshold
    data_filtered = data.loc[expression_ratio <= threshold]

    dropped = data.shape[0] - data_filtered.shape[0]
    if dropped > 0:
        print(
            f"  • Dropped {dropped} cells (high {filter_name}: >{threshold*100}%)")

    return data_filtered


def filter_doublets(data: pd.DataFrame, expected_doublet_rate: float = 0.05) -> pd.DataFrame:
    """
    Detects and removes doublets (cells that are actually two or more cells captured together).

    Uses Scrublet via scanpy to identify likely doublet events.
    Doublets are artifacts that can confound clustering and downstream analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Expression matrix with cells as rows and genes as columns.
    expected_doublet_rate : float, default=0.05
        Expected doublet rate of the dataset (typically 0.05-0.1 for 10x genomics).
        Can be estimated as: (number_of_cells_captured / number_of_cells_expected) - 1

    Returns
    -------
    data_filtered : pd.DataFrame
        DataFrame with doublets removed.

    Notes
    -----
    This function temporarily converts the DataFrame to an AnnData object for processing
    with scanpy, then converts back to a DataFrame.
    """
    # Convert DataFrame to AnnData for scanpy processing
    adata = sc.AnnData(data)

    # Run Scrublet doublet detection
    sc.pp.scrublet(adata, expected_doublet_rate=expected_doublet_rate)

    # Extract doublet predictions (cells marked as doublets have True in 'predicted_doublet' column)
    is_doublet = adata.obs["predicted_doublet"].values

    # Filter to keep only singlets (non-doublets)
    data_filtered = data.loc[~pd.Series(is_doublet, index=data.index)]

    dropped = data.shape[0] - data_filtered.shape[0]
    if dropped > 0:
        print(f"  • Dropped {dropped} doublets")
    else:
        print(f"  • No doublets detected")

    return data_filtered
