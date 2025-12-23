from typing import Union, List
import pandas as pd
from typing import cast
import warnings
import numpy as np
from .genes import APOPTOSIS_GENES, RRNA_GENES


def filter_low_magnitude_genes(data, min_count=2) -> pd.DataFrame:
    """
    Removes genes that never exceed a specific count threshold.
    (removes genes containing only 0s and 1s).

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

    print(
        f"[Filter Magnitude] Dropped {dropped} genes with max count < {min_count} (only 0s and 1s).")

    return data_filtered


def filter_high_mito_cells(data: pd.DataFrame, percentile=95) -> pd.DataFrame:
    """
    Removes cells with high mitochondrial expression (indicative of broken cells).
    """
    mt_genes = [gene for gene in data.columns if gene.upper().startswith("MT-")]

    print("[Filter Mito] Starting mitochondrial gene removal...")

    return filter_cells_by_fraction(
        data,
        gene_list=mt_genes,
        percentile=percentile,
    )


def filter_high_apoptosis_cells(data: pd.DataFrame, percentile=95) -> pd.DataFrame:
    """
    Removes cells with high expression of apoptosis-related genes (indicative of cell stress).
    """
    print("[Filter Apoptosis] Starting apoptosis gene removal...")

    return filter_cells_by_fraction(
        data,
        gene_list=APOPTOSIS_GENES,
        percentile=percentile,
    )


def filter_high_rrna_cells(data: pd.DataFrame, percentile=95) -> pd.DataFrame:
    """
    Removes cells with high rRNA expression (indicative of technical noise).
    """
    print("[Filter rRNA] Starting rRNA gene removal...")

    return filter_cells_by_fraction(
        data,
        gene_list=RRNA_GENES,
        percentile=percentile,
    )


def filter_cells_by_fraction(data: pd.DataFrame, gene_list: Union[List[str], np.ndarray], percentile: int) -> pd.DataFrame:
    """
    Removes cells with high expression of a specific gene set.
    """
    valid_genes = [gene for gene in gene_list if gene in data.columns]

    if len(valid_genes) == 0:
        warnings.warn(f"No genes found in dataset. Skipping.")
        return data

    # Calculate fraction: (sum of subset counts) / (total counts)
    subset_counts = data[valid_genes].sum(axis=1)
    total_counts = data.sum(axis=1)

    # Avoid division by zero by replacing 0 total counts with 1 (these cells will be dropped anyway or have 0 fraction)
    expression_ratio = subset_counts / total_counts.replace(0, 1)

    # Determine Cutoff
    cutoff = np.percentile(expression_ratio, percentile)

    # Keep cells where the ratio is LESS than the cutoff
    data_filtered = data.loc[expression_ratio < cutoff]

    initial_cells = data.shape[0]
    dropped = initial_cells - data_filtered.shape[0]

    print(f"Cutoff: {cutoff:.4f} (Ratio)")
    print(
        f"Dropped {dropped} cells (Top {100-percentile}% expression).")

    return cast(pd.DataFrame, data_filtered)
