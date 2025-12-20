import scprep
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


def filter_high_mito_cells(data, percentile=95) -> pd.DataFrame:
    """
    Identifies mitochondrial genes and removes cells with high mitochondrial expression.

    Mitochondrial genes usually start with "MT-" in humans or "mt-" in mice.

    Parameters
    ----------
    data : array-like
    percentile : int, default=95
        The percentile cutoff above which cells are removed.

    Returns
    -------
    data_filtered : array-like

    Example
    --------
    >>> data
              Gene_1  MT-Gene
    Sample_1    100       5   # Low Mito
    Sample_2     10     500   # High Mito (likely dead cell)
    Sample_3    100       5   # Low Mito

    >>> filter_high_mito_cells(data, percentile=95)
    # Sample_2 is removed
              Gene_1  MT-Gene
    Sample_1    100       5
    Sample_3    100       5
    """
    initial_cells = data.shape[0]

    # 1. Identify MT genes
    mt_genes = scprep.select.get_gene_set(data, starts_with="MT-")

    # STOP if still no genes found
    if len(mt_genes) == 0:
        warnings.warn(
            "[Filter Mito] No mitochondrial genes found (starting with 'MT-'). Skipping filtering.")
        return data

    # 2. Calculate expression
    # Formula: (sum of mito counts in cell) / (total counts in cell)

    mito_counts = data[mt_genes].sum(axis=1)
    total_counts = data.sum(axis=1)

    # Avoid division by zero
    mito_expression = mito_counts / total_counts.replace(0, 1)

    # 3. Filter cells
    cutoff = np.percentile(mito_expression, percentile)

    # Keep cells where mito_expression is LESS than the cutoff
    data_filtered = data.loc[mito_expression < cutoff]

    dropped = initial_cells - data_filtered.shape[0]
    print(f"[Filter Mito]  Cutoff: {cutoff:.4f}")
    print(
        f"[Filter Mito]  Dropped {dropped} cells (Top {100-percentile}% mitochondrial expr).")

    return cast(pd.DataFrame, data_filtered)


def filter_apoptosis_genes(data: pd.DataFrame) -> pd.DataFrame:
    """Removes apoptosis-related genes from the data."""
    print("[Filter Apoptosis] Starting apoptosis gene removal...")
    return filter_genes_by_name(data, genes_to_remove=APOPTOSIS_GENES)


def filter_rrna_genes(data: pd.DataFrame) -> pd.DataFrame:
    """Removes rRNA-related genes from the data."""
    print("[Filter rRNA] Starting rRNA gene removal...")
    return filter_genes_by_name(data, genes_to_remove=RRNA_GENES)


def filter_genes_by_name(data: pd.DataFrame, genes_to_remove: list[str]) -> pd.DataFrame:
    """
    Removes genes from the data based on a provided set of gene names.

    Parameters
    ----------
    data : pd.DataFrame, shape=[n_samples, n_features]
        The input single-cell data.
    genes_to_remove : set
        A set of gene names (columns) to be removed.

    Returns
    -------
    data_filtered : pd.DataFrame
        The data with the specified genes removed.
    """
    initial_genes = data.shape[1]

    # Convert the input list to a set for fast O(1) membership checking.
    genes_to_remove_set = set(genes_to_remove)

    # Iterate through the original column names (which are ordered)
    # and only keep a gene if it is NOT in the fast lookup set.
    genes_to_keep = [
        gene for gene in data.columns
        if gene not in genes_to_remove_set
    ]

    # Filter the DataFrame using the newly created, ordered list of genes to keep.
    data_filtered = data[genes_to_keep]

    dropped_count = initial_genes - data_filtered.shape[1]
    removed_count = len(genes_to_remove)

    print(
        f"[Filter Custom] Requested to remove {removed_count} genes. "
        f"Found and dropped {dropped_count} genes."
    )

    return data_filtered
