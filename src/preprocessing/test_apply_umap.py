import numpy as np
import pandas as pd
import pytest
from .apply_umap import apply_umap


@pytest.fixture
def pca_data():
    """Two well-separated groups of cells in a 5-component PCA space."""
    rng = np.random.default_rng(0)
    values = np.vstack([
        rng.normal(0, 1, size=(20, 5)),
        rng.normal(10, 1, size=(20, 5)),
    ])
    return pd.DataFrame(
        values,
        index=[f"Cell_{i}" for i in range(40)],
        columns=[f"PC{i+1}" for i in range(5)],
    )


def test_umap_returns_2d_with_cell_index(pca_data):
    umap_data = apply_umap(pca_data)

    assert umap_data.shape == (40, 2)
    assert list(umap_data.columns) == ["UMAP1", "UMAP2"]
    assert umap_data.index.equals(pca_data.index)


def test_umap_is_reproducible(pca_data):
    pd.testing.assert_frame_equal(apply_umap(pca_data), apply_umap(pca_data))
