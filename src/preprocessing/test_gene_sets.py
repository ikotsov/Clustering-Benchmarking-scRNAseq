import pandas as pd
from .gene_sets import (
    alias_matches,
    apoptosis_genes,
    duplicated_genes,
    ensembl_id,
    is_ensembl_id,
    matching_genes,
    mito_genes,
    rrna_genes,
    unmatched_genes,
)


ANNOTATION = pd.DataFrame({
    'gene_id': ['ENSG00000198804', 'ENSG00000000001', 'ENSG00000000002', 'ENSG00000075624'],
    'gene_name': ['MT-CO1', 'GENEA', 'GENEB', 'ACTB'],
    'alias_names': ['MTCO1;COX1', 'OLD1;SHARED', 'shared', 'NO_ALIAS'],
})


def test_is_ensembl_id_recognises_human_and_mouse_ids():
    assert is_ensembl_id('ENSG00000198804')
    assert is_ensembl_id('ENSMUSG00000064351.1')


def test_is_ensembl_id_rejects_gene_names():
    # ENSA is a human gene symbol; AC000061.1 is an older lncRNA name with a dot.
    assert not is_ensembl_id('ENSA')
    assert not is_ensembl_id('AC000061.1')


def test_ensembl_id_strips_version_and_uppercases():
    assert ensembl_id('ENSG00000198804.2') == 'ENSG00000198804'
    assert ensembl_id('ensmusg00000064351.1') == 'ENSMUSG00000064351'


def test_matching_genes_matches_names_and_ids_case_insensitively():
    data = pd.DataFrame(columns=['mt-co1', 'ENSG00000198804.2', 'ACTB'])

    assert matching_genes(data, ['MT-CO1', 'ENSG00000198804']) == [
        'mt-co1', 'ENSG00000198804.2']


def test_mito_genes_holds_names_and_ensembl_ids():
    human = mito_genes("human")
    mouse = mito_genes("mouse")

    assert 'MT-CO1' in human and 'ENSG00000198804' in human
    assert 'mt-Co1' in mouse
    assert any(gene.startswith('ENSMUSG') for gene in mouse)


def test_annotated_gene_sets_hold_names_and_ensembl_ids():
    assert 'BAX' in apoptosis_genes("human")
    assert 'ENSMUSG00000003873' in apoptosis_genes("mouse")  # Bax
    assert 'RNA5S1' in rrna_genes("human")
    assert any(gene.startswith('ENSG') for gene in rrna_genes("human"))



def test_matching_genes_does_not_treat_ens_prefixed_names_as_ids():
    data = pd.DataFrame(columns=['ENSA', 'ACTB'])

    assert matching_genes(data, ['ensa']) == ['ENSA']


def test_alias_matches_reports_every_annotation_row_an_alias_points_to():
    matches = alias_matches(['shared', 'cox1', 'UNKNOWN'], ANNOTATION)

    assert matches[['gene', 'n_matches', 'gene_name']].values.tolist() == [
        ['shared', 2, 'GENEA'],
        ['shared', 2, 'GENEB'],
        ['cox1', 1, 'MT-CO1'],
    ]


def test_alias_matches_ignores_the_no_alias_placeholder():
    assert alias_matches(['NO_ALIAS'], ANNOTATION).empty


def test_unmatched_genes_splits_unmatched_columns_by_cause():
    data = pd.DataFrame(columns=[
        'ENSG00000198804.2', 'actb', 'OLD1', 'ENSG00000999999', 'ERCC-00002'])

    unmatched = unmatched_genes(data, ANNOTATION)

    assert unmatched.alias_matches['gene'].tolist() == ['OLD1']
    assert unmatched.unknown_ensembl_ids == ['ENSG00000999999']
    assert unmatched.other == ['ERCC-00002']


def test_duplicated_genes_is_empty_when_gene_names_are_unique():
    data = pd.DataFrame([[1, 2]], columns=['GENEA', 'GENEB'])

    assert duplicated_genes(data).empty


def test_duplicated_genes_reports_repeats_and_whether_their_counts_match():
    data = pd.DataFrame(
        [[1, 1, 5, 6, 7, 0], [2, 2, 5, 0, 7, 0]],
        columns=['GENEA', 'GENEA', 'GENEB', 'GENEB', 'GENEB', 'GENEC'],
    )

    report = duplicated_genes(data)

    assert report.index.tolist() == ['GENEB', 'GENEA']
    assert report['n_columns'].tolist() == [3, 2]
    assert report['identical_counts'].tolist() == [False, True]
