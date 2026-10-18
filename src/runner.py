import logging
import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from src.data_loading import (
    load_csv_data,
    load_ground_truth_labels,
    load_tuned_params,
    parse_preprocessing_config,
)
from src.dataset_config import load_dataset_config, parse_n_clusters, parse_species
from src.preprocessing import apply_pca, preprocess_data
from src.preprocessing.gene_annotation import load_gene_annotation
from src.clustering.registry import (
    ALGORITHMS_REQUIRING_N_CLUSTERS,
    ClusteringAlgorithm,
    ClusteringFunc,
    get_clustering_strategy,
)
from src.evaluation import (
    evaluate_clustering_biologically,
    evaluate_clustering_externally,
    evaluate_clustering_internally,
    save_biological_evaluation_results,
    save_run_result,
)
from src.constants import DATASETS_DIR, PCA_VARIANCE_RATIO
from src.paths import get_pca_label, processed_data_path
from src.types import NormMethod, PathLike, Species
from src.evaluation.biological.types import EnrichmentSetName

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


def run_experiment(
    accession: str,
    algo_name: ClusteringAlgorithm,
    *,
    data_path: PathLike,
    config_path: PathLike,
    params_path: PathLike | None,
    labels_path: PathLike,
    result_path: PathLike,
    norm_method: NormMethod = "pearson",
    with_pca: bool = True,
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
        Tuning result of this algorithm + normalization; None runs the algorithm with its defaults
    labels_path : PathLike
        Ground truth labels CSV
    result_path : PathLike
        Where to save this run's evaluation metrics (collected into results.tsv by aggregate_run_results)
    norm_method : NormMethod, default="pearson"
        Normalization method ("pearson" or "log_cpm")
    """
    pca_tag = get_pca_label(with_pca).upper()
    logger.info("EXPERIMENT: %s + %s + %s",
                accession, algo_name.upper(), pca_tag)

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

    labels = cluster_func(target_data, **cluster_kwargs)

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
        metrics=metrics,
        output_path=result_path,
    )


def _format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A (undefined)"


def run_biological_evaluation(
    accession: str,
    algo_name: ClusteringAlgorithm,
    norm_method: NormMethod,
    with_pca: bool,
    ground_truth: pd.Series,
    species: Species,
    clustering_strategy: ClusteringFunc,
    clustering_kwargs: Mapping[str, object] | None = None,
    enrichment_set_names: list[EnrichmentSetName] | None = None,
):
    """Run the biological evaluation stage and persist the results."""
    dataset_dir = os.path.join(DATASETS_DIR, accession)

    logger.info("Biological evaluation...")
    biological_input_path = processed_data_path(
        accession, norm_method, with_pca=False)
    if not os.path.exists(biological_input_path):
        raise FileNotFoundError(
            f"Biological input not found: {biological_input_path}"
        )

    biological_data = load_csv_data(biological_input_path)

    biological_results = evaluate_clustering_biologically(
        expression_data=biological_data,
        cell_type_labels=ground_truth.astype(str),
        clustering_strategy=clustering_strategy,
        clustering_kwargs=clustering_kwargs,
        enrichment_set_names=enrichment_set_names,
        species=species,
    )
    logger.info("Biological comparisons retained: %s rows",
                len(biological_results))

    output_dir = os.path.join(dataset_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    save_biological_evaluation_results(
        dataset=accession,
        algorithm=algo_name,
        preprocessing=norm_method,
        with_pca=with_pca,
        comparison_records=biological_results,
        output_dir=output_dir,
    )
