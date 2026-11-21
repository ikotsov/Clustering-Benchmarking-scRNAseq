import numpy as np
import pandas as pd
import pytest

from .sampling import build_matched_gene_samples, compute_gene_sampling_features


def test_compute_gene_sampling_features_uses_log_scaled_raw_counts():
    counts = pd.DataFrame({"A": [0, 3, 0, 1], "B": [0, 0, 0, 0]})

    features = compute_gene_sampling_features(counts)

    assert features.loc["A", "log_expressing_cells"] == pytest.approx(np.log1p(2))
    assert features.loc["A", "log_mean_count"] == pytest.approx(np.log1p(1.0))
    assert features.loc["B"].tolist() == pytest.approx([0.0, 0.0])


def _features(n_genes: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(n_genes, 2)),
        index=[f"G{i}" for i in range(n_genes)],
        columns=["log_expressing_cells", "log_mean_count"],
    )


def test_build_matched_gene_samples_draws_new_sets_of_the_observed_size():
    features = _features(200)
    observed = features.index[:20].tolist()

    samples = build_matched_gene_samples(features, observed, n_samples=5, seed=3)

    assert len(samples) == 5
    assert all(len(sample) == 20 and len(set(sample)) == 20 for sample in samples)
    assert all(set(sample) <= set(features.index) for sample in samples)
    assert any(set(sample) != set(observed) for sample in samples)
    assert samples == build_matched_gene_samples(features, observed, n_samples=5, seed=3)


def test_build_matched_gene_samples_rejects_a_universe_no_larger_than_the_observed_set():
    # The original bug: sampling all HVGs from a universe of only the HVGs.
    features = _features(10)

    with pytest.raises(ValueError, match="whole universe"):
        build_matched_gene_samples(features, features.index.tolist(), n_samples=2)
