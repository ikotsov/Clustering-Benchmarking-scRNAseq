import pandas as pd
import numpy as np
import pytest
from .transforms import (
    normalize_by_library_size,
    log_transform,
    normalize_with_log_cpm,
    normalize_with_pearson,
    select_hvgs,
)


@pytest.fixture
def sample_data():
    """A simple count matrix for testing normalization and log transform."""
    data = {
        'Gene_A': [50, 50],
        'Gene_B': [50, 150],
        'Gene_Zero': [0, 0]  # Useful for checking log(0) behavior
    }
    return pd.DataFrame(data, index=['Cell_1', 'Cell_2'])


def test_normalization_sums_to_1M(sample_data):
    rescale_factor = 1_000_000
    normalized_data = normalize_by_library_size(
        sample_data,
        rescale=rescale_factor
    )

    # Cell_1 had 50 counts for Gene_A out of 100 total.
    # (50 / 100) * 1,000,000 = 500,000
    assert normalized_data.loc['Cell_1', 'Gene_A'] == 500_000.0


def test_normalization_sums_to_10K(sample_data):
    rescale_factor = 10_000
    normalized_data = normalize_by_library_size(
        sample_data,
        rescale=rescale_factor
    )

    # Cell_1 Gene_A: (50 / 100) * 10,000 = 5,000.0
    assert normalized_data.loc['Cell_1', 'Gene_A'] == 5_000.0


def test_normalization_sum(sample_data):
    rescale_factor = 1_000_000
    normalized_data = normalize_by_library_size(
        sample_data,
        rescale=rescale_factor
    )

    # Check that the sum of every row equals the rescale factor
    row_sums = normalized_data.sum(axis=1)
    is_equal = (row_sums == rescale_factor).all()
    assert is_equal, f"Rows did not sum exactly to {rescale_factor}.\nActual sums:\n{row_sums}"


def test_log_transform_values(sample_data):
    transformed_data = log_transform(sample_data, pseudocount=1)

    assert (transformed_data['Gene_Zero'] == 0.0).all()
    expected_value = np.log(50 + 1)
    actual_value = transformed_data.loc['Cell_2', 'Gene_A']
    assert actual_value == pytest.approx(expected_value)


def test_log_transform_custom_pseudocount(sample_data):
    transformed_data = log_transform(sample_data, pseudocount=5)

    expected_value = np.log(50 + 5)
    actual_value = transformed_data.loc['Cell_2', 'Gene_A']
    assert actual_value == pytest.approx(expected_value)


def test_normalization_handles_zero_total_count_row():
    # A cell with 0 counts across all genes would otherwise divide by zero.
    df_zero_row = pd.DataFrame({
        'Gene_A': [0, 50],
        'Gene_B': [0, 50],
    }, index=['Empty_Cell', 'Normal_Cell'])

    normalized_data = normalize_by_library_size(df_zero_row)

    assert not normalized_data.isna().any().any()
    assert (normalized_data.loc['Empty_Cell'] == 0.0).all()


def test_normalize_with_log_cpm_matches_manual_composition(sample_data):
    combined = normalize_with_log_cpm(sample_data)
    manual = log_transform(normalize_by_library_size(sample_data))

    pd.testing.assert_frame_equal(combined, manual)


@pytest.fixture
def count_matrix():
    """
    A larger, variable synthetic count matrix. HVG selection and Pearson
    residuals need enough cells/genes and expression variability to avoid
    numerical errors in scanpy's underlying fits.
    """
    rng = np.random.default_rng(0)
    n_cells, n_genes = 50, 20
    gene_means = rng.uniform(1, 50, size=n_genes)
    counts = rng.poisson(gene_means, size=(n_cells, n_genes))
    return pd.DataFrame(
        counts,
        index=[f'Cell_{i}' for i in range(n_cells)],
        columns=[f'Gene_{i}' for i in range(n_genes)],
    )


def test_select_hvgs_log_cpm_returns_subset_of_genes(count_matrix):
    hvgs = select_hvgs(count_matrix, norm_method="log_cpm")

    assert len(hvgs) > 0
    assert set(hvgs).issubset(set(count_matrix.columns))


def test_select_hvgs_pearson_returns_subset_of_genes(count_matrix):
    hvgs = select_hvgs(count_matrix, norm_method="pearson")

    assert len(hvgs) > 0
    assert set(hvgs).issubset(set(count_matrix.columns))


def test_select_hvgs_unknown_method_raises(count_matrix):
    with pytest.raises(ValueError):
        select_hvgs(count_matrix, norm_method="bogus_method")


def test_normalize_with_pearson_preserves_shape(count_matrix):
    residuals = normalize_with_pearson(count_matrix)

    assert residuals.shape == count_matrix.shape
    assert not residuals.isna().any().any()
