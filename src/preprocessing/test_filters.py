import pandas as pd
import pytest
from . import filters as filters_module
from .gene_annotation import load_gene_annotation
from .filters import (
    filter_genes_by_max_count,
    filter_cells_by_fraction,
    filter_high_mito_cells,
    filter_high_apoptosis_cells,
    filter_rrna_genes,
    filter_doublets,
    high_fraction_mask,
)


def test_filter_genes_by_max_count():
    data = pd.DataFrame({
        'High_Expr': [10, 20, 0],   # Keep: Max is 20
        'Low_Expr':  [1,  1,  1],   # Drop: Only 1s (popular, but low signal)
        'Binary':    [0,  1,  0],   # Drop: Only 0s and 1s
        'Zero':      [0,  0,  0]    # Drop: Only 0s
    }, index=['C1', 'C2', 'C3'])

    filtered = filter_genes_by_max_count(data)

    # Only 'High_Expr' should remain
    assert list(filtered.columns) == ['High_Expr']
    assert filtered.shape == (3, 1)


def test_filter_keeps_rare_but_high_genes():
    data = pd.DataFrame({
        # Present in only 1 cell, but count is 100
        'Rare_But_Strong': [100, 0, 0],
    }, index=['C1', 'C2', 'C3'])

    filtered = filter_genes_by_max_count(data)

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
        threshold=0.5
    )

    assert 'Cell_Bad' not in filtered.index
    assert 'Cell_Border' in filtered.index
    assert 'Cell_Good' in filtered.index


def test_fraction_handles_missing_genes(fraction_data):
    filtered = filter_cells_by_fraction(
        fraction_data,
        gene_list=['Target_1', 'Ghost_Gene'],
        threshold=0.5
    )

    assert 'Cell_Bad' not in filtered.index
    assert filtered.shape[0] == 2


def test_fraction_handles_zero_total_counts():
    df_zero = pd.DataFrame({
        'Gene_A': [0, 10],
        'Gene_B': [0, 10]
    }, index=['Zero_Cell', 'Normal_Cell'])

    filtered = filter_cells_by_fraction(
        df_zero,
        gene_list=['Gene_A'],
        threshold=0.1
    )

    assert 'Zero_Cell' in filtered.index


def test_fraction_matches_genes_case_insensitively(fraction_data):
    # Gene list casing differs from the data columns (e.g. mouse list on uppercased data).
    filtered = filter_cells_by_fraction(
        fraction_data,
        gene_list=['target_1'],
        threshold=0.5
    )

    assert 'Cell_Bad' not in filtered.index
    assert filtered.shape[0] == 2


def test_high_fraction_mask_flags_cells_above_threshold(fraction_data):
    mask = high_fraction_mask(fraction_data, ['Target_1'], threshold=0.5)

    assert mask.to_dict() == {
        'Cell_Bad': True, 'Cell_Border': False, 'Cell_Good': False}


def test_high_fraction_mask_without_matching_genes_flags_nothing(fraction_data):
    mask = high_fraction_mask(fraction_data, ['Ghost_Gene'], threshold=0.5)

    assert not mask.any()
    assert mask.index.equals(fraction_data.index)


def test_mito_wrapper_finds_mt_genes():
    df_mito = pd.DataFrame({
        'MT-CO1': [100, 0],  # 100% Mito in Cell 1
        'ACTB':   [0, 100]   # 0% Mito in Cell 2
    }, index=['High_Mito', 'Low_Mito'])

    filtered = filter_high_mito_cells(df_mito, threshold=0.05)

    assert 'High_Mito' not in filtered.index
    assert 'Low_Mito' in filtered.index


def test_mito_wrapper_is_case_insensitive():
    # Lowercase "mt-" prefix is common in mouse datasets.
    df_mito = pd.DataFrame({
        'mt-co1': [100, 0],
        'Actb':   [0, 100]
    }, index=['High_Mito', 'Low_Mito'])

    filtered = filter_high_mito_cells(df_mito, threshold=0.05)

    assert 'High_Mito' not in filtered.index
    assert 'Low_Mito' in filtered.index


def test_fraction_filtering_empty_gene_list_is_noop(fraction_data):
    filtered = filter_cells_by_fraction(
        fraction_data,
        gene_list=[],
        threshold=0.5
    )

    pd.testing.assert_frame_equal(filtered, fraction_data)


def test_apoptosis_wrapper_filters_high_expression():
    df_apoptosis = pd.DataFrame({
        'BAX': [100, 0],   # 100% apoptosis-gene expression in Cell 1
        'ACTB': [0, 100]
    }, index=['High_Apoptosis', 'Low_Apoptosis'])

    filtered = filter_high_apoptosis_cells(
        df_apoptosis, threshold=0.05, species="human")

    assert 'High_Apoptosis' not in filtered.index
    assert 'Low_Apoptosis' in filtered.index


def test_filter_rrna_genes_removes_rrna_columns():
    df_rrna = pd.DataFrame({
        'RNA5S1': [100, 0],   # rRNA
        'MT-RNR2': [5, 5],    # mitochondrial rRNA
        'ACTB':   [0, 100]
    }, index=['C1', 'C2'])

    filtered = filter_rrna_genes(df_rrna, species="human")

    assert list(filtered.columns) == ['ACTB']
    assert filtered.index.equals(df_rrna.index)


def test_filter_rrna_genes_handles_mouse_names():
    df_rrna = pd.DataFrame({
        'Rn5s': [100, 0],
        'Actb': [0, 100]
    }, index=['C1', 'C2'])

    filtered = filter_rrna_genes(df_rrna, species="mouse")

    assert list(filtered.columns) == ['Actb']


def test_filter_doublets_removes_predicted_doublets(monkeypatch):
    """
    Mocks scanpy's Scrublet call so the test exercises only
    filter_doublets' own filtering logic, not Scrublet's stochastic
    detection algorithm.
    """
    data = pd.DataFrame({
        'Gene_A': [10, 20, 30, 40],
        'Gene_B': [5, 15, 25, 35],
    }, index=['C1', 'C2', 'C3', 'C4'])

    def fake_scrublet(adata, expected_doublet_rate=0.05, threshold=None):
        adata.obs['predicted_doublet'] = [False, True, False, True]

    monkeypatch.setattr(filters_module.sc.pp, 'scrublet', fake_scrublet)
    monkeypatch.setattr(
        filters_module.sc.pl, 'scrublet_score_distribution', lambda adata: None)

    filtered = filter_doublets(data)

    assert list(filtered.index) == ['C1', 'C3']


def test_mito_wrapper_matches_ensembl_ids():
    # ENSG00000198804 is MT-CO1; the version suffix must not prevent the match.
    df_mito = pd.DataFrame({
        'ENSG00000198804.2': [100, 0],
        'ENSG00000075624':   [0, 100]   # ACTB
    }, index=['High_Mito', 'Low_Mito'])

    filtered = filter_high_mito_cells(df_mito, threshold=0.05, species="human")

    assert 'High_Mito' not in filtered.index
    assert 'Low_Mito' in filtered.index


def test_apoptosis_wrapper_matches_mouse_ensembl_ids():
    df_apoptosis = pd.DataFrame({
        'ENSMUSG00000003873': [100, 0],  # Bax
        'ENSMUSG00000029580': [0, 100]   # Actb
    }, index=['High_Apoptosis', 'Low_Apoptosis'])

    filtered = filter_high_apoptosis_cells(
        df_apoptosis, threshold=0.05, species="mouse")

    assert 'High_Apoptosis' not in filtered.index
    assert 'Low_Apoptosis' in filtered.index


def test_filter_rrna_genes_matches_ensembl_ids():
    rrna_id = load_gene_annotation("human").set_index(
        "gene_name").loc["RNA5S1", "gene_id"]
    df_rrna = pd.DataFrame({
        f"{rrna_id}.1": [100, 0],
        'ENSG00000075624': [0, 100]   # ACTB
    }, index=['C1', 'C2'])

    filtered = filter_rrna_genes(df_rrna, species="human")

    assert list(filtered.columns) == ['ENSG00000075624']
