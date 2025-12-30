import numpy as np
import pandas as pd
from typing import cast
import scanpy as sc


CPM_RESCALE = 1_000_000


def normalize_by_library_size(data: pd.DataFrame, rescale: int = CPM_RESCALE) -> pd.DataFrame:
    """
    Normalizes counts per cell to sum to `rescale` (default CPM).

    Parameters
    ----------
    data : pd.DataFrame
    rescale : int, default=1_000_000

    Returns
    -------
    data_norm : pd.DataFrame

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
        f"[Normalize] Normalizing library size (CPM) with rescale={rescale:.0e}...")

    # Calculate the sum of counts for each cell (row)
    library_size = data.sum(axis=1)
    # Divide each row by its sum and multiply by the rescale factor
    data_norm = data.div(library_size, axis=0) * rescale

    # Fill NaNs with 0 in case a cell had 0 total counts
    return data_norm.fillna(0)


def log_transform(data: pd.DataFrame, pseudocount: int = 1) -> pd.DataFrame:
    """
    Applies log transformation: log(x + pseudocount).

    Parameters
    ----------
    data : pd.DataFrame
    pseudocount : int, default=1

    Returns
    -------
    data_log : pd.DataFrame

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
    print(f"[Transform] Applying log transform (log{pseudocount}+x)...")

    data_log = np.log10(data + pseudocount)

    # Cast because np.log10 is typed to return an ndarray,
    # but it returns a DataFrame when the input is a DataFrame.
    return cast(pd.DataFrame, data_log)


def normalize_data_with_pearson(filtered_data: pd.DataFrame, n_hvg: int = 3000) -> pd.DataFrame:
    """
    Computes analytic Pearson Residuals (SCTransform equivalent) using Scanpy.
    """
    print(
        f"[Residuals] Computing Pearson residuals for {filtered_data.shape[0]} cells using Scanpy...")

    # Setup AnnData
    adata = sc.AnnData(filtered_data)

    # Select Highly Variable Genes (HVGs)
    # Reference: https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html
    print(f"Selecting top {n_hvg} variable genes...")
    sc.experimental.pp.highly_variable_genes(
        adata,
        flavor="pearson_residuals",
        n_top_genes=n_hvg
    )

    # Compute Pearson Residuals
    # Reference: https://scanpy.readthedocs.io/en/stable/generated/scanpy.experimental.pp.normalize_pearson_residuals.html
    # This updates adata.X in-place with the residuals.
    print("Calculating residuals...")
    sc.experimental.pp.normalize_pearson_residuals(adata)

    # Subset to HVGs and Export
    # We filter the AnnData object to only keep the genes we selected previously.
    adata_hvg = adata[:, adata.var["highly_variable"]]

    # Convert to DataFrame
    # Scanpy's .to_df() handles the conversion from sparse matrix to DataFrame automatically.
    pearson_df = adata_hvg.to_df()

    return pearson_df
