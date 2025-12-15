import pandas as pd
import numpy as np
import pytest
from .filters import filter_genes_by_name


@pytest.fixture
def sample_data():
    """A minimal, reproducible DataFrame for testing."""
    data = {
        'Gene_A': [10, 20, 30],
        'Gene_B': [1, 2, 3],
        'MT-Gene': [100, 200, 300],
        'RPS18': [5, 5, 5],
        'Gene_C': [40, 50, 60]
    }

    # Create the DataFrame with clear index names
    df = pd.DataFrame(data, index=['Cell_1', 'Cell_2', 'Cell_3'])
    return df


def test_genes_are_correctly_removed(sample_data):
    genes_to_exclude = ["Gene_B", "RPS18"]
    expected_genes_after_filter = ['Gene_A', 'MT-Gene', 'Gene_C']

    filtered_data = filter_genes_by_name(sample_data, genes_to_exclude)

    # Check the final columns match the expected set
    assert list(filtered_data.columns) == expected_genes_after_filter
    # Check the number of genes is correct
    assert filtered_data.shape[1] == sample_data.shape[1] - \
        len(genes_to_exclude)

    # Check that the data values of the kept genes are unchanged
    # (We check that 'Gene_A' column in both DFs is identical)
    pd.testing.assert_series_equal(
        sample_data['Gene_A'],
        filtered_data['Gene_A']
    )


def test_non_existent_genes_are_handled(sample_data):
    genes_to_exclude = ["Gene_B", "NON_EXISTENT_GENE"]
    # Only Gene_B should be removed
    expected_genes_after_filter = ['Gene_A', 'MT-Gene', 'RPS18', 'Gene_C']

    filtered_data = filter_genes_by_name(sample_data, genes_to_exclude)

    # The final column count should only reflect the *one* gene that was actually present ('Gene_B')
    assert filtered_data.shape[1] == sample_data.shape[1] - 1
    assert list(filtered_data.columns) == expected_genes_after_filter


def test_empty_gene_list(sample_data):
    genes_to_exclude = []

    filtered_data = filter_genes_by_name(sample_data, genes_to_exclude)

    # The resulting DataFrame should be identical to the input
    pd.testing.assert_frame_equal(sample_data, filtered_data)
    assert filtered_data.shape == sample_data.shape
