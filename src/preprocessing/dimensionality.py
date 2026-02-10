import pandas as pd
from sklearn.decomposition import PCA

from src.constants import SEED


def apply_pca(data: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
    """
    Apply PCA dimensionality reduction using scikit-learn.

    Parameters
    ----------
    data : pd.DataFrame
        Normalized gene expression data (cells × genes)
    n_components : int, default=50
        Number of principal components to retain

    Returns
    -------
    pca_data : pd.DataFrame
        PCA-transformed data (cells × components)

    Examples
    --------
    >>> normalized_data.shape
    (124, 3000)

    >>> pca_data = apply_pca(normalized_data, n_components=50)
    >>> pca_data.shape
    (124, 50)
    """
    print(f"  • Applying PCA (reducing to {n_components} components)")

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=SEED)
    pca_result = pca.fit_transform(data)

    # Convert back to DataFrame with proper column names
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_data = pd.DataFrame(
        pca_result,
        index=data.index,
        columns=pc_columns
    )

    # Report explained variance
    total_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"  • Explained variance: {total_variance:.1f}%")

    return pca_data
