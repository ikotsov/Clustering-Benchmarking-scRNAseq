import logging

import numpy as np
import pandas as pd
from typing import cast
import scanpy as sc


logger = logging.getLogger(__name__)

# Note: Seurat's LogNormalize scales to 10_000 (CP10k) by default.
# We use CPM (1_000_000)
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
    logger.debug("Normalizing library size (rescale=%0.0e)", rescale)

    # Calculate the sum of counts for each cell (row)
    library_size = data.sum(axis=1)
    # Divide each row by its sum and multiply by the rescale factor
    data_norm = data.div(library_size, axis=0) * rescale

    # Fill NaNs with 0 in case a cell had 0 total counts
    return data_norm.fillna(0)


def log_transform(data: pd.DataFrame, pseudocount: int = 1) -> pd.DataFrame:
    """
    Applies natural log transformation: ln(x + pseudocount).

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
    # ln(100 + 1) ~= 4.62, ln(0 + 1) = 0
                Gene_A     Gene_B
    Sample_1  4.615121        0.0
    """
    logger.debug("Applying log transform (ln(x+%s))", pseudocount)

    data_log = np.log(data + pseudocount)

    # Cast because np.log is typed to return an ndarray,
    # but it returns a DataFrame when the input is a DataFrame.
    return cast(pd.DataFrame, data_log)


# In Seurat, 3,000 HVGs is the default for Pearson residuals,
# and 2,000 for log-CPM.
PEARSON_N_HVG = 3_000
LOG_CPM_N_HVG = 2_000


def select_hvgs_seurat_v3(filtered_data: pd.DataFrame) -> list[str]:
    """
    Selects Highly Variable Genes (HVGs) with the Seurat v3 flavor, used with log-CPM.
    Expects raw counts.
    """
    logger.debug("Selecting top %s variable genes (Seurat v3)", LOG_CPM_N_HVG)

    adata = sc.AnnData(filtered_data)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3_paper",
        n_top_genes=LOG_CPM_N_HVG,
    )

    return adata.var_names[adata.var["highly_variable"]].tolist()


def select_hvgs_pearson(filtered_data: pd.DataFrame) -> list[str]:
    """
    Selects Highly Variable Genes (HVGs) with the Pearson residuals flavor.
    Expects raw counts.
    """
    logger.debug(
        "Selecting top %s variable genes (Pearson residuals)", PEARSON_N_HVG)

    adata = sc.AnnData(filtered_data)
    # Reference: https://scanpy.readthedocs.io/en/stable/generated/scanpy.experimental.pp.highly_variable_genes.html
    sc.experimental.pp.highly_variable_genes(
        adata,
        flavor="pearson_residuals",
        n_top_genes=PEARSON_N_HVG,
    )

    return adata.var_names[adata.var["highly_variable"]].tolist()


def normalize_with_pearson(hvg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes analytic Pearson Residuals (sctransform equivalent) using Scanpy.
    Follows: https://scanpy.readthedocs.io/en/latest/tutorials/experimental/pearson_residuals.html
    """
    logger.debug("Computing residuals for %s cells", hvg_data.shape[0])

    adata = sc.AnnData(hvg_data)

    # Reference: https://scanpy.readthedocs.io/en/stable/generated/scanpy.experimental.pp.normalize_pearson_residuals.html
    # This updates adata.X in-place with the residuals.
    sc.experimental.pp.normalize_pearson_residuals(adata)

    # Scanpy's .to_df() handles the conversion from sparse matrix to DataFrame automatically.
    return adata.to_df()


def normalize_with_log_cpm(filtered_data: pd.DataFrame) -> pd.DataFrame:
    """
    Receives the full filtered matrix (not an HVG subset) so library sizes 
    reflect sequencing depth.
    """
    data = normalize_by_library_size(filtered_data)
    data = log_transform(data)

    return data
