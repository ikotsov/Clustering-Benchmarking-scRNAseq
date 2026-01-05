import pandas as pd
import pytest
from .utils import select_hvg_by_variance  # Adjust import path accordingly


@pytest.fixture
def hvg_data():
    """
    Gene_High_Var: Variance is high (0, 50, 100)
    Gene_Mid_Var: Variance is medium (0, 5, 10)
    Gene_Low_Var: Variance is low (1, 1, 1)
    Gene_Const: Variance is zero (10, 10, 10)
    """
    return pd.DataFrame({
        'Gene_High_Var': [0, 50, 100],
        'Gene_Mid_Var':  [0, 5, 10],
        'Gene_Low_Var':  [1, 1.1, 1.2],
        'Gene_Const':    [10, 10, 10]
    }, index=['C1', 'C2', 'C3'])


def test_select_hvg_by_fixed_count(hvg_data):
    filtered = select_hvg_by_variance(hvg_data, n_top_genes=2)

    assert list(filtered.columns) == ['Gene_High_Var', 'Gene_Mid_Var']
    assert filtered.shape[1] == 2


def test_select_hvg_by_percentile(hvg_data):
    filtered = select_hvg_by_variance(hvg_data, percentile=0.25)

    # 4 genes total, 25% should be 1 gene
    assert list(filtered.columns) == ['Gene_High_Var']
    assert filtered.shape[1] == 1


def test_select_hvg_default_fallback(hvg_data):
    filtered = select_hvg_by_variance(hvg_data)

    # Default n_top_genes is 2000, so all genes should be returned
    assert filtered.shape[1] == 4


def test_select_hvg_preserves_cell_order(hvg_data):
    filtered = select_hvg_by_variance(hvg_data, n_top_genes=1)

    # Ensure indices (cells) are not shuffled
    assert list(filtered.index) == ['C1', 'C2', 'C3']


def test_select_hvg_handles_all_zero_variance():
    # All genes are constant
    df_const = pd.DataFrame({
        'G1': [1, 1, 1],
        'G2': [5, 5, 5]
    })

    # Should still return the requested number of genes (even if variance is tied at 0)
    filtered = select_hvg_by_variance(df_const, n_top_genes=1)
    assert filtered.shape[1] == 1
