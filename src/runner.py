import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from src.data_loading import (
    load_csv_data,
    load_ground_truth_labels,
    load_tuned_params,
    parse_preprocessing_config,
)
from src.dataset_config import load_dataset_config, parse_marker_databases, parse_n_clusters, parse_species
from src.preprocessing import apply_pca, apply_umap, preprocess_data
from src.preprocessing.gene_annotation import load_gene_annotation
from src.preprocessing.pipeline import run_filtering_pipeline
from src.clustering.registry import (
    ALGORITHMS_REQUIRING_N_CLUSTERS,
    ClusteringAlgorithm,
    get_clustering_strategy,
)
from src.evaluation import (
    compute_null_distribution,
    evaluate_clustering_biologically,
    evaluate_clustering_externally,
    evaluate_clustering_internally,
    save_null_distribution,
    save_run_result,
)
from src.constants import DATASETS_DIR, PCA_VARIANCE_RATIO, SEED
from src.paths import get_pca_label, processed_data_path
from src.profiling import measure_resources
from src.tuning.objectives import get_params_label
from src.tuning.types import ObjectiveMetric
from src.types import NormMethod, PathLike, Species
from src.evaluation.biological import (
    build_reference_sets,
    load_gene_marker_index,
    load_reference_sets,
    save_biological_evaluation,
    save_reference_sets,
)

logger = logging.getLogger(__name__)


def run_preprocessing(
    *,
    raw_path: PathLike,
    config_path: PathLike,
    annotation_path: PathLike,
    pca_output_path: PathLike,
    no_pca_output_path: PathLike,
    hvg_output_path: PathLike,
    norm_method: NormMethod = "pearson",
    pca_variance_ratio: float = PCA_VARIANCE_RATIO,
):
    """
    Runs only the preprocessing step and saves the result.
    Useful for exploring preprocessing outputs or running multiple preprocessing strategies for all datasets at once.

    Parameters
    ----------
    raw_path : PathLike
        Raw counts matrix (cells x genes, gzipped CSV)
    config_path : PathLike
        Dataset config.yaml (species and QC thresholds)
    annotation_path : PathLike
        Gene annotation table of the dataset's species
    pca_output_path, no_pca_output_path : PathLike
        Where to save the PCA and non-PCA representations (gzipped CSV)
    hvg_output_path : PathLike
        Where to save the selected HVG names (TSV)
    norm_method : NormMethod, default="pearson"
        Normalization method ("pearson" or "log_cpm")
    pca_variance_ratio : float, default=PCA_VARIANCE_RATIO
        Fraction of variance to preserve when PCA is applied
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Expected file not found: {raw_path}")

    config = load_dataset_config(config_path)
    species = parse_species(config)
    preprocessing_config = parse_preprocessing_config(config)
    annotation = load_gene_annotation(species, annotation_path)

    # Load & preprocess
    logger.info(
        "PREPROCESSING: %s, norm_method=%s, pca_variance_ratio=%.0f%%",
        raw_path,
        norm_method,
        pca_variance_ratio * 100,
    )
    raw_data = load_csv_data(raw_path)

    logger.info("Building non-PCA representation...")
    preprocessed_no_pca, hvg_genes_no_pca = preprocess_data(
        raw_data,
        norm_method=norm_method,
        species=species,
        preprocessing_config=preprocessing_config,
        annotation=annotation,
    )

    logger.info("Building PCA representation from non-PCA data...")
    preprocessed_pca = apply_pca(
        preprocessed_no_pca,
        variance_ratio=pca_variance_ratio,
    )

    # Save both representations
    for preprocessed_data, save_path in (
        (preprocessed_pca, pca_output_path),
        (preprocessed_no_pca, no_pca_output_path),
    ):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        preprocessed_data.to_csv(save_path, compression='gzip')
        logger.info(
            "Saved to: %s (%s × %s features)",
            save_path,
            preprocessed_data.shape[0],
            preprocessed_data.shape[1],
        )

    Path(hvg_output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.Series(hvg_genes_no_pca, name="gene").to_csv(
        hvg_output_path,
        sep="\t",
        index=False,
    )
    logger.info("Saved HVGs to: %s (%s genes)",
                hvg_output_path, len(hvg_genes_no_pca))


def run_umap(
    *,
    pca_path: PathLike,
    output_path: PathLike,
):
    """
    Embeds the PCA representation in 2D with UMAP and saves the coordinates.
    The embedding is for visualization only; it is never clustered or scored.

    Parameters
    ----------
    pca_path : PathLike
        PCA representation (output of run_preprocessing)
    output_path : PathLike
        Where to save the UMAP coordinates (gzipped CSV)
    """
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"Expected file not found: {pca_path}")

    logger.info("UMAP: %s", pca_path)
    pca_data = load_csv_data(pca_path)
    umap_data = apply_umap(pca_data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    umap_data.to_csv(output_path, compression='gzip')
    logger.info("Saved to: %s (%s cells)", output_path, umap_data.shape[0])


def run_experiment(
    accession: str,
    algo_name: ClusteringAlgorithm,
    *,
    data_path: PathLike,
    config_path: PathLike,
    params_path: PathLike | None,
    labels_path: PathLike,
    result_path: PathLike,
    labels_output_path: PathLike,
    norm_method: NormMethod = "pearson",
    with_pca: bool = True,
    tuning_metric: ObjectiveMetric | None = None,
):
    """
    Orchestrates the clustering flow: Load preprocessed data -> Cluster -> Evaluate -> Save.
    Computes the external and internal metrics only; the biological evaluation is a separate step
    (run_biological_evaluation).
    Expects preprocessed data to already exist (use run_preprocessing first).
    PCA is applied during preprocessing.

    Parameters
    ----------
    accession : str
        Dataset accession ID (recorded in the results)
    algo_name : str
        Clustering algorithm name
    data_path : PathLike
        Preprocessed data to cluster (output of run_preprocessing)
    config_path : PathLike
        Dataset config.yaml (n_clusters)
    params_path : PathLike or None
        Tuning result of this algorithm + normalization + PCA option; None runs the algorithm with its defaults
    labels_path : PathLike
        Ground truth labels CSV
    result_path : PathLike
        Where to save this run's evaluation metrics (collected into results.tsv by aggregate_run_results)
    labels_output_path : PathLike
        Where to save the predicted cluster of each cell (CSV), e.g. for plotting on the UMAP
    norm_method : NormMethod, default="pearson"
        Normalization method ("pearson" or "log_cpm")
    tuning_metric : ObjectiveMetric or None
        Metric params_path was tuned on (e.g. "silhouette"); None, like params_path, for the defaults
    """
    if (params_path is None) != (tuning_metric is None):
        raise ValueError("params_path and tuning_metric must be given together")
    pca_tag = get_pca_label(with_pca).upper()
    logger.info("EXPERIMENT: %s + %s + %s + %s",
                accession, algo_name.upper(), pca_tag, get_params_label(tuning_metric).upper())

    # 2. Load preprocessed data
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed file not found: {data_path}\n"
            "Run preprocessing first with run_preprocessing."
        )

    logger.info("Loading preprocessed data: %s", data_path)
    target_data = load_csv_data(data_path)

    # 3. Clustering
    config = load_dataset_config(config_path)
    logger.info("Clustering (%s)...", algo_name)
    cluster_func = get_clustering_strategy(algo_name)

    cluster_kwargs = load_tuned_params(params_path)

    n_clusters = parse_n_clusters(config)
    if algo_name in ALGORITHMS_REQUIRING_N_CLUSTERS and n_clusters is not None:
        cluster_kwargs["n_clusters"] = n_clusters

    with measure_resources() as clustering_resources:
        labels = cluster_func(target_data, **cluster_kwargs)
    logger.info(
        "Clustering took %.2fs (CPU %.2fs), memory %.0f MB (+%.0f MB)",
        clustering_resources.wall_time_s,
        clustering_resources.cpu_time_s,
        clustering_resources.rss_mb,
        clustering_resources.rss_increase_mb,
    )

    # 4. Load ground truth labels
    logger.info("Loading ground truth labels...")
    ground_truth = load_ground_truth_labels(labels_path)
    logger.info("Loaded %s ground truth labels", len(ground_truth))

    # 5. Evaluate and save results
    logger.info("Evaluation...")
    # Here we align the predicted labels with the ground truth labels based on the index (cell IDs).
    # The alignment is done based on the order of the indices in the target_data.
    # Wew rely on the fact that the clustering algorithm returns labels in the same order as the input data.
    labels_series = pd.Series(
        labels, index=target_data.index, name="cluster")
    Path(labels_output_path).parent.mkdir(parents=True, exist_ok=True)
    labels_series.to_csv(labels_output_path)
    logger.info("Saved predicted labels to: %s", labels_output_path)

    external_metrics = evaluate_clustering_externally(
        labels_series, ground_truth)

    logger.info("ARI: %.3f", external_metrics["ari"])
    logger.info("NMI: %.3f", external_metrics["nmi"])
    logger.info("Jaccard: %.3f", external_metrics["jaccard"])

    internal_metrics = evaluate_clustering_internally(
        target_data, labels_series)
    logger.info("Silhouette: %s", _format_metric(
        internal_metrics['silhouette']))
    logger.info("Calinski-Harabasz: %s",
                _format_metric(internal_metrics['calinski_harabasz']))
    logger.info("Davies-Bouldin: %s",
                _format_metric(internal_metrics['davies_bouldin']))

    metrics = {**external_metrics, **internal_metrics}

    save_run_result(
        algorithm=algo_name,
        preprocessing=norm_method,
        with_pca=with_pca,
        tuning_metric=tuning_metric,
        metrics=metrics,
        output_path=result_path,
        n_cells=target_data.shape[0],
        n_features=target_data.shape[1],
        clustering_resources=clustering_resources.as_dict(),
    )


def run_permutation_test(
    *,
    data_path: PathLike,
    run_labels_path: PathLike,
    labels_path: PathLike,
    n_permutations: int,
    output_path: PathLike,
    seed: int = SEED,
):
    """
    Builds the null distribution of one benchmark run's metrics by shuffling its predicted labels
    n_permutations times (see compute_null_distribution) and saves it.
    Expects the run to already exist (use run_experiment first).

    Parameters
    ----------
    data_path : PathLike
        Preprocessed data the run clustered (scored by the internal metrics)
    run_labels_path : PathLike
        Predicted cluster of each cell (labels_output_path of run_experiment)
    labels_path : PathLike
        Ground truth labels CSV
    n_permutations : int
        Number of shuffles of the predicted labels
    output_path : PathLike
        Where to save the null distribution (.npz)
    seed : int, default=SEED
        Seed of the shuffles, so reruns give the same null distribution
    """
    logger.info("PERMUTATION TEST: %s (%s permutations, seed=%s)", run_labels_path, n_permutations, seed)
    data = load_csv_data(data_path)
    labels_pred = pd.read_csv(run_labels_path, index_col=0).iloc[:, 0]
    ground_truth = load_ground_truth_labels(labels_path)

    null = compute_null_distribution(data, labels_pred, ground_truth, n_permutations=n_permutations, seed=seed)
    null_means = {metric: float(np.mean(null[metric].to_numpy())) for metric in null.columns}
    logger.info("Null means: %s", ", ".join(f"{m}={_format_metric(v)}" for m, v in null_means.items()))
    save_null_distribution(output_path, null, seed=seed)


def _format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A (undefined)"


def _load_filtered_counts(raw_path: PathLike, config_path: PathLike, annotation_path: PathLike):
    """The raw counts after the cell QC and gene filtering of preprocessing (all genes, not only HVGs)."""
    config = load_dataset_config(config_path)
    species = parse_species(config)
    annotation = load_gene_annotation(species, annotation_path)
    counts = run_filtering_pipeline(
        load_csv_data(raw_path), config=parse_preprocessing_config(config), species=species, annotation=annotation)
    return counts, config, species, annotation


def run_reference_sets(
    *,
    raw_path: PathLike,
    config_path: PathLike,
    annotation_path: PathLike,
    labels_path: PathLike,
    output_path: PathLike,
    marker_index_path: PathLike | None = None,
):
    """
    Builds the reference sets of a dataset (see build_reference_sets) and saves them: the DE markers of each
    ground-truth cell type, on log-CPM of all filtered genes, plus the marker databases set in the dataset's
    config.yaml (see parse_marker_databases). They are the same for every run, so they are built once per dataset.

    Parameters
    ----------
    raw_path, config_path, annotation_path : PathLike
        Raw counts, dataset config.yaml and gene annotation, filtered as in preprocessing
    labels_path : PathLike
        Ground truth labels CSV
    output_path : PathLike
        Where to save the reference sets (JSON)
    marker_index_path : PathLike or None
        Gene marker index (scripts/generate_gene_marker_index.py); needed only when the config sets marker databases
    """
    counts, config, species, annotation = _load_filtered_counts(raw_path, config_path, annotation_path)
    marker_databases = parse_marker_databases(config)
    logger.info("REFERENCE SETS: %s (%s cells x %s genes), de_markers%s", raw_path, counts.shape[0], counts.shape[1],
                "".join(f" + {name}" for name in marker_databases))

    reference_sets = build_reference_sets(
        counts=counts,
        cell_type_labels=load_ground_truth_labels(labels_path),
        species=species,
        annotation=annotation,
        marker_databases=marker_databases,
        marker_index=load_gene_marker_index(marker_index_path) if marker_databases and marker_index_path else None,
    )
    save_reference_sets(output_path, reference_sets)


def run_biological_evaluation(
    *,
    raw_path: PathLike,
    config_path: PathLike,
    annotation_path: PathLike,
    run_labels_path: PathLike,
    reference_sets_path: PathLike,
    output_path: PathLike,
    n_samples: int,
    min_marker_genes: int,
    seed: int = SEED,
):
    """
    Biological evaluation of one benchmark run (see evaluate_clustering_biologically): do the marker genes of the
    run's clusters match the reference sets' genes more than expression-matched random gene sets do?
    Expects the run (run_experiment) and the dataset's reference sets (run_reference_sets) to already exist;
    the run's saved clusters are evaluated, nothing is clustered again.

    Parameters
    ----------
    raw_path, config_path, annotation_path : PathLike
        Raw counts, dataset config.yaml and gene annotation, filtered as in preprocessing
    run_labels_path : PathLike
        Predicted cluster of each cell (labels_output_path of run_experiment)
    reference_sets_path : PathLike
        The dataset's reference sets (output of run_reference_sets)
    output_path : PathLike
        Where to save the evaluation (JSON); its summary becomes the bio_* columns of results.tsv
    n_samples : int
        Random gene sets per cluster
    min_marker_genes : int
        Clusters with fewer marker genes are scored but not tested
    seed : int, default=SEED
        Seed of the random sets, so reruns give the same null distributions
    """
    logger.info("BIOLOGICAL EVALUATION: %s (%s random sets per cluster, seed=%s)", run_labels_path, n_samples, seed)
    counts, _, _, _ = _load_filtered_counts(raw_path, config_path, annotation_path)
    predicted_labels = pd.read_csv(run_labels_path, index_col=0).iloc[:, 0]

    evaluation = evaluate_clustering_biologically(
        counts=counts,
        predicted_labels=predicted_labels,
        reference_sets=load_reference_sets(reference_sets_path),
        n_samples=n_samples,
        seed=seed,
        min_marker_genes=min_marker_genes,
    )
    logger.info("Summary: %s", ", ".join(f"{name}={value}" for name, value in evaluation.summary.items()))
    save_biological_evaluation(output_path, evaluation, n_samples=n_samples, seed=seed)
