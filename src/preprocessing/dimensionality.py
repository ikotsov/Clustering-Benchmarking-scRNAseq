import pandas as pd
from sklearn.decomposition import PCA

from src.constants import SEED, PCA_VARIANCE_RATIO


def apply_pca(data: pd.DataFrame, variance_ratio: float = PCA_VARIANCE_RATIO) -> pd.DataFrame:
    """
    Apply PCA dimensionality reduction using scikit-learn.

    Parameters
    ----------
    data : pd.DataFrame
        Normalized gene expression data (cells × genes)
    variance_ratio : float, default=0.80
        Fraction of total variance to preserve (must be in (0, 1))

    Returns
    -------
    pca_data : pd.DataFrame
        PCA-transformed data (cells × components)

    Examples
    --------
    >>> normalized_data.shape
    (124, 3000)

    >>> pca_data = apply_pca(normalized_data, variance_ratio=0.80)
    >>> pca_data.shape[0]
    124
    """
    if not 0 < variance_ratio < 1:
        raise ValueError(
            f"variance_ratio must be in (0, 1), got {variance_ratio}")

    print(
        f"  • Applying PCA (target explained variance: {variance_ratio:.0%})")

    # Fit PCA
    pca = PCA(n_components=variance_ratio, random_state=SEED)
    pca_result = pca.fit_transform(data)

    # Convert back to DataFrame with proper column names
    n_components_used = pca_result.shape[1]
    pc_columns = [f"PC{i+1}" for i in range(n_components_used)]
    pca_data = pd.DataFrame(
        pca_result,
        index=data.index,
        columns=pc_columns
    )

    # Report explained variance
    total_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"  • Retained {n_components_used} components")
    print(f"  • Explained variance: {total_variance:.1f}%")

    return pca_data
