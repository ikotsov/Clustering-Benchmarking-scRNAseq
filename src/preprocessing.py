import scprep
import pandas as pd
from typing import cast
import warnings
import numpy as np


def preprocess_sc_data(raw_data, min_cells=1, mito_percentile=95) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline.
    """
    print(f"--- Starting Preprocessing: Input Shape {raw_data.shape} ---")

    data = filter_rare_genes(raw_data, min_cells=min_cells)
    # First remove dead cells, because dead cells distort the library size normalization for the healthy cells.
    data = filter_high_mito_cells(data, percentile=mito_percentile)
    data = normalize_by_library_size(data)
    data = log_transform(data)

    print(f"--- Finished Preprocessing: Final Shape {data.shape} ---")
    return data


def filter_rare_genes(data, min_cells=1) -> pd.DataFrame:
    """
    Removes genes expressed in fewer than `min_cells` cells (drops zero-expression genes).

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
    min_cells : int, default=1

    Returns
    -------
    data_filtered : array-like

    Examples
    --------
    >>> data
              Gene_1  Gene_2  Gene_Zero
    Sample_1     10      20         0
    Sample_2     10      20         0
    Sample_3     10      20         0

    >>> filter_rare_genes(data, min_cells=1)
    # Gene_Zero is removed because it is detected in 0 cells
              Gene_1  Gene_2
    Sample_1     10      20
    Sample_2     10      20
    Sample_3     10      20
    """
    initial_genes = data.shape[1]

    data_filtered = scprep.filter.filter_rare_genes(
        data, cutoff=0, min_cells=min_cells
    )

    dropped = initial_genes - data_filtered.shape[1]
    print(
        f"[Filter Genes] Dropped {dropped} genes expressed in < {min_cells} cells.")

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

    Examples
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
            "[Filter Mito] No mitochondrial genes found (starting with 'MT-' or 'mt-'). Skipping filtering.")
        return data

    # 2. Calculate Expression Manually (Robust)
    # Formula: (Sum of Mito Counts in cell) / (Total Counts in cell)

    # .sum(axis=1) works reliably on both Sparse and Dense DataFrames
    mito_counts = data[mt_genes].sum(axis=1)
    total_counts = data.sum(axis=1)

    # Avoid division by zero
    mito_expression = mito_counts / total_counts.replace(0, 1)

    # 3. Filter using Pandas
    cutoff = np.percentile(mito_expression, percentile)

    # Keep cells where mito_expression is LESS than the cutoff
    data_filtered = data.loc[mito_expression < cutoff]

    dropped = initial_cells - data_filtered.shape[0]
    print(f"[Filter Mito]  Cutoff: {cutoff:.4f}")
    print(
        f"[Filter Mito]  Dropped {dropped} cells (Top {100-percentile}% mitochondrial expr).")

    return cast(pd.DataFrame, data_filtered)


def normalize_by_library_size(data, rescale=1_000_000) -> pd.DataFrame:
    """
    Normalizes counts per cell to sum to `rescale` (default CPM).

    If data is sparse, pseudocount must be 1 such that log(0 + pseudocount) = 0

    Parameters
    ----------
    data : array-like
    rescale : int, default=1_000_000

    Returns
    -------
    data_norm : array-like

    Examples
    --------
    >>> data (Total counts: S1=1000, S2=2000)
              Gene_A  Gene_B
    Sample_1     500     500
    Sample_2    1000    1000

    >>> normalize_library_size(data, rescale=1_000_000)
    # Both samples normalized to same depth
                 Gene_A     Gene_B
    Sample_1   500000.0   500000.0
    Sample_2   500000.0   500000.0
    """
    print(
        f"[Normalize]   Normalizing library size (CPM) with rescale={rescale:.0e}...")
    return scprep.normalize.library_size_normalize(data, rescale=rescale)


def log_transform(data, pseudocount=1) -> pd.DataFrame:
    """
    Applies log transformation: log(x + pseudocount).

    Parameters
    ----------
    data : array-like
    pseudocount : int, default=1

    Returns
    -------
    data_log : array-like

    Examples
    --------
    >>> data (CPM)
              Gene_A     Gene_B
    Sample_1   100.0        0.0

    >>> log_transform(data, pseudocount=1)
    # log(100 + 1) ~= 4.61, log(0 + 1) = 0
                Gene_A     Gene_B
    Sample_1  4.615121        0.0
    """
    print(f"[Transform]   Applying log transform (log{pseudocount}+x)...")

    data_log = scprep.transform.log(data, pseudocount=pseudocount)

    # Cast to DataFrame since scprep returns a generic array-like
    return cast(pd.DataFrame, data_log)
