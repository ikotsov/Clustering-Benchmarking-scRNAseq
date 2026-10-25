import math

import numpy as np
import pandas as pd
import pytest

from .permutation import (
    NULL_METRICS,
    compute_null_distribution,
    load_null_distribution,
    permute_labels,
    save_null_distribution,
    summarize_null,
)


def _blobs(n_per_cluster: int = 20, n_clusters: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    cells = pd.Index([f"cell_{i}" for i in range(n_per_cluster * n_clusters)])
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    X = rng.normal(scale=0.1, size=(len(cells), 2)) + labels[:, None] * 10
    return pd.DataFrame(X, index=cells), pd.Series(labels, index=cells)


def test_permute_labels_keeps_cluster_sizes_and_is_reproducible():
    labels = np.array([0, 0, 0, 1, 1, 2])

    permuted = permute_labels(labels, n_permutations=50, seed=1)

    assert permuted.shape == (50, 6)
    assert all(sorted(row) == sorted(labels) for row in permuted)
    assert np.array_equal(permuted, permute_labels(labels, n_permutations=50, seed=1))
    assert not np.array_equal(permuted, permute_labels(labels, n_permutations=50, seed=2))


def test_perfect_clustering_beats_every_shuffle():
    data, labels = _blobs()
    n_permutations = 99

    null = compute_null_distribution(data, labels, labels, n_permutations=n_permutations, seed=0)
    summary = summarize_null({"ari": 1.0, "nmi": 1.0, "davies_bouldin": 0.01}, null)

    assert list(null.columns) == NULL_METRICS
    assert len(null) == n_permutations
    assert abs(null["ari"].mean()) < 0.05
    assert summary["ari_pvalue"] == pytest.approx(1 / (n_permutations + 1))
    assert summary["nmi_pvalue"] == pytest.approx(1 / (n_permutations + 1))
    # Lower is better for Davies-Bouldin: a tiny observed score beats every shuffle.
    assert summary["davies_bouldin_pvalue"] == pytest.approx(1 / (n_permutations + 1))


def test_summarize_null_counts_ties_and_skips_undefined_metrics():
    null = pd.DataFrame({"ari": [0.0, 0.5, 1.0], "silhouette": [math.nan] * 3})

    summary = summarize_null({"ari": 0.5, "silhouette": None}, null)

    assert summary["ari_null_mean"] == pytest.approx(0.5)
    assert summary["ari_pvalue"] == pytest.approx(3 / 4)
    assert math.isnan(summary["silhouette_null_mean"])
    assert math.isnan(summary["silhouette_pvalue"])


def test_internal_metrics_are_nan_for_a_single_cluster():
    data, _ = _blobs()
    one_cluster = pd.Series(0, index=data.index)

    null = compute_null_distribution(data, one_cluster, one_cluster, n_permutations=5, seed=0)

    assert null[["silhouette", "calinski_harabasz", "davies_bouldin"]].isna().to_numpy().all()


def test_null_distribution_round_trip(tmp_path):
    null = pd.DataFrame(np.arange(12, dtype=float).reshape(2, 6), columns=pd.Index(NULL_METRICS))
    path = tmp_path / "null.npz"

    save_null_distribution(path, null, seed=3)

    pd.testing.assert_frame_equal(load_null_distribution(path), null)
