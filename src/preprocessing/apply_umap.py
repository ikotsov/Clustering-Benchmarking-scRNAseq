import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from src.constants import SEED


logger = logging.getLogger(__name__)


def apply_umap(pca_data: pd.DataFrame) -> pd.DataFrame:
    """
    Embed the PCA representation in 2D with UMAP, using scanpy's defaults, for visualization only.
    UMAP distorts distances and densities, so the embedding is never clustered or scored.

    Parameters
    ----------
    pca_data : pd.DataFrame
        PCA representation (cells × components)

    Returns
    -------
    umap_data : pd.DataFrame
        UMAP coordinates (cells × [UMAP1, UMAP2])

    Examples
    --------
    >>> pca_data.shape
    (124, 30)

    >>> apply_umap(pca_data).shape
    (124, 2)
    """
    logger.debug("Applying UMAP")

    adata = ad.AnnData(pca_data.to_numpy(dtype=float))
    # use_rep="X": the input is already PCA, so scanpy must not run its own PCA on it (it does with >= 50 columns)
    sc.pp.neighbors(adata, use_rep="X", random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)

    return pd.DataFrame(
        np.asarray(adata.obsm["X_umap"]),
        index=pca_data.index,
        columns=["UMAP1", "UMAP2"],
    )
