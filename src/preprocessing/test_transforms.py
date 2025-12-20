import pandas as pd
import numpy as np
import pytest
from .transforms import normalize_by_library_size, log_transform


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
    expected_value = np.log10(50 + 1)
    actual_value = transformed_data.loc['Cell_2', 'Gene_A']
    assert actual_value == pytest.approx(expected_value)
