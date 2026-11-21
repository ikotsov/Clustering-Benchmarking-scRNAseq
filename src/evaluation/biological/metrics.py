"""Metrics of a cluster's marker genes vs the reference sets, and their null statistics.

For one cluster, the marker set M, every random set (same size as M) and each reference set R
live in the same universe U, so every metric is a function of a single number, the overlap |M∩R|.
The null test therefore only needs the overlaps of the random sets: the p-value of the overlap
is the p-value of every metric (they all rise with it, FPR falls with it).
"""
import numpy as np
import pandas as pd

from .constants import SIGNIFICANCE_LEVEL
from .types import ReferenceSets

# Columns of overlap_metrics and null_statistics, in order
METRIC_NAMES = ("tp", "fp", "fn", "tn", "jaccard", "precision", "recall", "specificity", "fpr",
                "expected", "fold_enrichment")
NULL_STATISTIC_NAMES = ("null_overlap_mean", "overlap_z_score", "p_value")


def count_overlaps(genes: set[str] | list[str], reference_genes: list[set[str]]) -> np.ndarray:
    """|genes ∩ R| for each reference set R."""
    gene_set = set(genes)
    return np.array([len(gene_set & reference) for reference in reference_genes])


def overlap_metrics(
    overlap: np.ndarray,
    n_marker_genes: int,
    reference_sizes: np.ndarray,
    universe_size: int,
) -> dict[str, np.ndarray]:
    """Set metrics of a marker set (prediction) vs each reference set (truth), from their overlaps.

    Assumes a non-empty marker set and non-empty reference sets smaller than the universe,
    as evaluate_clustering_biologically guarantees.
    """
    tp = overlap
    fp = n_marker_genes - tp
    fn = reference_sizes - tp
    tn = universe_size - n_marker_genes - reference_sizes + tp
    expected = n_marker_genes * reference_sizes / universe_size
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "jaccard": tp / (tp + fp + fn),
        "precision": tp / n_marker_genes,
        "recall": tp / reference_sizes,
        "specificity": tn / (tn + fp),
        "fpr": fp / (tn + fp),
        # Overlap expected for a uniformly random set of the same size
        "expected": expected,
        "fold_enrichment": tp / expected,
    }


def null_statistics(observed: np.ndarray, random_overlaps: np.ndarray) -> dict[str, np.ndarray]:
    """Test each observed overlap against the overlaps of the random sets.

    observed: one overlap per reference set; random_overlaps: random sets x reference sets.
    - null_overlap_mean: overlap expected for an expression-matched random set
    - overlap_z_score: (observed - null mean) / null std; NaN when the null is constant
    - p_value: (#random >= observed + 1) / (#random + 1), so never 0
    """
    null_mean = random_overlaps.mean(axis=0)
    null_std = random_overlaps.std(axis=0)  # exactly 0 for a constant column: overlaps are integers
    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = np.where(null_std > 0, (observed - null_mean) / null_std, np.nan)
    n_as_large = (random_overlaps >= observed).sum(axis=0)
    return {
        "null_overlap_mean": null_mean,
        "overlap_z_score": z_score,
        "p_value": (n_as_large + 1) / (len(random_overlaps) + 1),
    }


def untested_statistics() -> dict[str, float]:
    """The null_statistics columns of a cluster that is not tested."""
    return dict.fromkeys(NULL_STATISTIC_NAMES, np.nan)


def summarize_biological_evaluation(
    records: pd.DataFrame,
    reference_sets: ReferenceSets,
    n_clusters: int,
    significance_level: float = SIGNIFICANCE_LEVEL,
) -> dict[str, float | int]:
    """One value per results-table column for a run.

    For each cell type of each reference set, its best-matching cluster is the one with the highest
    Jaccard; a cell type no cluster has markers for scores 0. Per reference set (macro average over
    its cell types, as in the reference study):

    - bio_<set>_jaccard / bio_<set>_fold_enrichment: mean of the best clusters' values
    - bio_<set>_n_recovered: cell types whose best cluster's p-value < significance_level
    - bio_<set>_n_cell_types: cell types of the reference set

    Plus bio_n_clusters, bio_n_clusters_without_markers (no records) and bio_n_clusters_untested
    (markers, but no null distribution; see null_status).
    """
    tested = records["null_status"] == "tested"
    summary: dict[str, float | int] = {
        "bio_n_clusters": n_clusters,
        "bio_n_clusters_without_markers": n_clusters - records["cluster_id"].nunique(),
        "bio_n_clusters_untested": records.loc[~tested, "cluster_id"].nunique(),
    }

    best = records.loc[records.groupby(["enrichment_set", "cell_type"])["jaccard"].idxmax()]
    for set_name, cell_types in reference_sets.items():
        best_of_set = (best[best["enrichment_set"] == set_name]
                       .set_index("cell_type")
                       .reindex(list(cell_types))[["jaccard", "fold_enrichment", "p_value"]]
                       .astype(float))
        summary[f"bio_{set_name}_jaccard"] = float(best_of_set["jaccard"].fillna(0).mean())
        summary[f"bio_{set_name}_fold_enrichment"] = float(best_of_set["fold_enrichment"].fillna(0).mean())
        summary[f"bio_{set_name}_n_recovered"] = int((best_of_set["p_value"] < significance_level).sum())
        summary[f"bio_{set_name}_n_cell_types"] = len(cell_types)
    return summary

