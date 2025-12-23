import pandas as pd
import pytest
from .filters import filter_low_magnitude_genes, filter_cells_by_fraction, filter_high_mito_cells


def test_filter_low_magnitude_genes():
    data = pd.DataFrame({
        'High_Expr': [10, 20, 0],   # Keep: Max is 20
        'Low_Expr':  [1,  1,  1],   # Drop: Only 1s (popular, but low signal)
        'Binary':    [0,  1,  0],   # Drop: Only 0s and 1s
        'Zero':      [0,  0,  0]    # Drop: Only 0s
    }, index=['C1', 'C2', 'C3'])

    filtered = filter_low_magnitude_genes(data)

    # Only 'High_Expr' should remain
    assert list(filtered.columns) == ['High_Expr']
    assert filtered.shape == (3, 1)


def test_filter_keeps_rare_but_high_genes():
    data = pd.DataFrame({
        # Present in only 1 cell, but count is 100
        'Rare_But_Strong': [100, 0, 0],
    }, index=['C1', 'C2', 'C3'])

    filtered = filter_low_magnitude_genes(data)

    # Should stay because 100 > 1
    assert 'Rare_But_Strong' in filtered.columns


@pytest.fixture
def fraction_data():
    """
    Data designed for easy percentage math.
    Cell_1: 100% Target Genes (Bad)
    Cell_2: 50% Target Genes (Borderline)
    Cell_3: 0% Target Genes (Good)
    """
    df = pd.DataFrame({
        'Target_1': [100, 50, 0],
        'Other_1':  [0,  50, 100],
    }, index=['Cell_Bad', 'Cell_Border', 'Cell_Good'])
    return df


def test_fraction_filtering_math(fraction_data):
    filtered = filter_cells_by_fraction(
        fraction_data,
        gene_list=['Target_1'],
        percentile=67  # Cutoff roughly at 67%, should drop Cell_Bad (100%)
    )

    assert 'Cell_Bad' not in filtered.index
    assert 'Cell_Border' in filtered.index
    assert 'Cell_Good' in filtered.index


def test_fraction_handles_missing_genes(fraction_data):
    filtered = filter_cells_by_fraction(
        fraction_data,
        gene_list=['Target_1', 'Ghost_Gene'],
        percentile=67
    )

    assert 'Cell_Bad' not in filtered.index
    assert filtered.shape[0] == 2


def test_fraction_handles_zero_total_counts():
    df_zero = pd.DataFrame({
        'Gene_A': [0, 10],
        'Gene_B': [0, 10]
    }, index=['Zero_Cell', 'Normal_Cell'])

    # Should not crash
    filtered = filter_cells_by_fraction(
        df_zero, gene_list=['Gene_A'], percentile=50)

    # Zero cell has 0/1 fraction = 0.0, so it is usually kept (unless cutoff is 0)
    assert 'Zero_Cell' in filtered.index


def test_mito_wrapper_finds_mt_genes():
    df_mito = pd.DataFrame({
        'MT-CO1': [100, 0],  # 100% Mito in Cell 1
        'ACTB':   [0, 100]   # 0% Mito in Cell 2
    }, index=['High_Mito', 'Low_Mito'])

    # Filter top 50% (should drop the High_Mito cell)
    filtered = filter_high_mito_cells(df_mito, percentile=50)

    assert 'High_Mito' not in filtered.index
    assert 'Low_Mito' in filtered.index
