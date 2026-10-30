import pandas as pd
from .qc_metrics import cell_qc_metrics, gene_set_fraction, gene_set_size


def test_gene_set_fraction_per_cell():
    data = pd.DataFrame({
        'Target_1': [100, 50, 0],
        'Other_1':  [0,  50, 100],
    }, index=['Cell_Bad', 'Cell_Border', 'Cell_Good'])

    fractions = gene_set_fraction(data, ['target_1'])

    assert fractions.to_dict() == {
        'Cell_Bad': 1.0, 'Cell_Border': 0.5, 'Cell_Good': 0.0}


def test_gene_set_fraction_without_matching_genes_is_zero():
    data = pd.DataFrame({'Gene_A': [10, 20]}, index=['C1', 'C2'])

    fractions = gene_set_fraction(data, ['Ghost_Gene'])

    assert fractions.to_dict() == {'C1': 0.0, 'C2': 0.0}


def test_gene_set_fraction_warning_names_the_gene_set(caplog):
    data = pd.DataFrame({'Gene_A': [10, 20]}, index=['C1', 'C2'])

    gene_set_fraction(data, ['Ghost_Gene'], name="rRNA")

    assert "None of the 1 rRNA genes" in caplog.text


def test_gene_set_size_counts_genes_listed_by_name_and_id_once():
    gene_list = ['Rn5s', 'n-R5s1', 'n-R5s1', 'ENSMUSG00000119584', 'ENSMUSG00000064339.1', 'ENSMUSG00000064339']

    assert gene_set_size(gene_list) == 2
    assert gene_set_size(['Ghost_Gene', 'Other_Gene']) == 2


def test_cell_qc_metrics_per_cell():
    data = pd.DataFrame({
        'MT-CO1': [50, 0, 10],
        'ACTB':   [50, 0, 0],
        'GAPDH':  [100, 30, 0],
    }, index=['Deep', 'Empty_Mito', 'Shallow'])

    metrics = cell_qc_metrics(data, species="human")

    assert list(metrics.index) == ['Deep', 'Empty_Mito', 'Shallow']
    assert metrics['count_depth'].to_dict() == {'Deep': 200, 'Empty_Mito': 30, 'Shallow': 10}
    assert metrics['n_genes'].to_dict() == {'Deep': 3, 'Empty_Mito': 1, 'Shallow': 1}
    assert metrics['mito_fraction'].to_dict() == {'Deep': 0.25, 'Empty_Mito': 0.0, 'Shallow': 1.0}
    assert set(metrics.columns) == {
        'count_depth', 'n_genes', 'mito_fraction', 'rrna_fraction', 'apoptosis_fraction'}
