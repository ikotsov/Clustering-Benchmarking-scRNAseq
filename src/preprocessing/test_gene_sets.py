import pandas as pd
from .gene_sets import apoptosis_genes, ensembl_id, is_ensembl_id, matching_genes, mito_genes, rrna_genes


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
