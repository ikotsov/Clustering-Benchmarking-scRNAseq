import logging
import os
from collections.abc import Mapping

import pandas as pd
from src.data_loading import (
    load_csv_data,
    load_clustering_params,
    load_dataset_config,
    load_ground_truth_labels,
    parse_n_clusters,
    parse_preprocessing_config,
)
from src.preprocessing import apply_pca, preprocess_data
from src.clustering.registry import ClusteringAlgorithm, ClusteringFunc, get_clustering_strategy
from src.evaluation import (
    evaluate_clustering_biologically,
    evaluate_clustering_externally,
    evaluate_clustering_internally,
    save_biological_evaluation_results,
    save_evaluation_results,
)
from src.constants import PCA_VARIANCE_RATIO
from src.types import NormMethod, Species
from src.utils import get_pca_label
from src.evaluation.biological.types import EnrichmentSetName

logger = logging.getLogger(__name__)


VALID_SPECIES: tuple[Species, Species] = ("human", "mouse")
ALGORITHMS_REQUIRING_N_CLUSTERS: tuple[ClusteringAlgorithm, ...] = (
    "agglomerative",
    "birch",
    "kmeans",
    "spectral",
)


def run_preprocessing(accession: str, norm_method: NormMethod = "pearson", pca_variance_ratio: float = PCA_VARIANCE_RATIO):
    """
    Runs only the preprocessing step and saves the result.
    Useful for exploring preprocessing outputs or running multiple preprocessing strategies for all datasets at once.

    Parameters
    ----------
    accession : str
        Dataset accession ID
    norm_method : NormMethod, default="pearson"
        Normalization method ("pearson" or "log_cpm")
    pca_variance_ratio : float, default=PCA_VARIANCE_RATIO
        Fraction of variance to preserve when PCA is applied
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "data", accession)

    raw_file_path = os.path.join(dataset_dir, "raw", f"{accession}.csv.gz")

    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Expected file not found: {raw_file_path}")

    config = load_dataset_config(dataset_dir)
    species_value = config.get("species")
    species = species_value if species_value in VALID_SPECIES else "human"
    preprocessing_config = parse_preprocessing_config(config)

    # Load & preprocess
    logger.info(
        "PREPROCESSING: %s, norm_method=%s, pca_variance_ratio=%s",
        accession,
        norm_method,
        f"{pca_variance_ratio:.0%}",
    )
    raw_data = load_csv_data(raw_file_path)

    logger.info("--- Building non-PCA representation ---")
    preprocessed_no_pca, hvg_genes_no_pca = preprocess_data(
        raw_data,
        norm_method=norm_method,
        species=species,
        preprocessing_config=preprocessing_config,
    )

    logger.info("--- Building PCA representation from non-PCA data ---")
    preprocessed_pca = apply_pca(
        preprocessed_no_pca,
        variance_ratio=pca_variance_ratio,
    )

    # Save both representations
    output_dir = os.path.join(dataset_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    for with_pca in (True, False):
        preprocessed_data = preprocessed_pca if with_pca else preprocessed_no_pca
        filename = _processed_filename(norm_method, with_pca)
        save_path = os.path.join(output_dir, filename)
        preprocessed_data.to_csv(save_path, compression='gzip')
        logger.info(
            "Saved to: %s (%s × %s features)",
            filename,
            preprocessed_data.shape[0],
            preprocessed_data.shape[1],
        )

    hvg_filename = _hvg_filename(norm_method)
    hvg_path = os.path.join(output_dir, hvg_filename)
    pd.Series(hvg_genes_no_pca, name="gene").to_csv(
        hvg_path,
        sep="\t",
        index=False,
    )
    logger.info("Saved HVGs to: %s (%s genes)",
                hvg_filename, len(hvg_genes_no_pca))


def run_experiment(
    accession: str,
    algo_name: ClusteringAlgorithm,
    norm_method: NormMethod = "pearson",
    with_pca: bool = True,
    enrichment_set_names: list[EnrichmentSetName] | None = None,
):
    """
    Orchestrates the clustering flow: Load preprocessed data -> Cluster -> Evaluate -> Save.
    Expects preprocessed data to already exist (use run_preprocessing first).
    PCA is applied during preprocessing.

    Parameters
    ----------
    accession : str
        Dataset accession ID
    algo_name : str
        Clustering algorithm name
    norm_method : NormMethod, default="pearson"
        Normalization method ("pearson" or "log_cpm")
    """
    # 1. Dynamic path resolution
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "data", accession)

    pca_tag = get_pca_label(with_pca).upper()
    logger.info("EXPERIMENT: %s + %s + %s",
                accession, algo_name.upper(), pca_tag)

    # 2. Load preprocessed data
    preprocessed_filename = _processed_filename(norm_method, with_pca)
    preprocessed_file = os.path.join(
        dataset_dir, "processed", preprocessed_filename)

    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(
            f"Preprocessed file not found: {preprocessed_filename}\n"
            f"Run preprocessing first with: run_preprocessing('{accession}', '{norm_method}')"
        )

    logger.info("Loading preprocessed data: %s", preprocessed_filename)
    target_data = load_csv_data(preprocessed_file)

    # 3. Clustering
    config = load_dataset_config(dataset_dir)
    species_value = config.get("species")
    species = species_value if species_value in VALID_SPECIES else "human"
    logger.info("Clustering (%s)...", algo_name)
    cluster_func = get_clustering_strategy(algo_name)

    cluster_kwargs = load_clustering_params(
        dataset_dir=dataset_dir,
        norm_method=norm_method,
        algorithm=algo_name,
    )

    n_clusters = parse_n_clusters(config)
    if algo_name in ALGORITHMS_REQUIRING_N_CLUSTERS and n_clusters is not None:
        cluster_kwargs["n_clusters"] = n_clusters

    labels = cluster_func(target_data, **cluster_kwargs)

    # 4. Load ground truth labels
    logger.info("Loading ground truth labels...")
    try:
        ground_truth = load_ground_truth_labels(dataset_dir)
        logger.info("Loaded %s ground truth labels", len(ground_truth))
    except FileNotFoundError as e:
        logger.warning("Warning: %s", e)
        logger.warning("Skipping evaluation.")
        return

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
    logger.info("Silhouette: %.3f", internal_metrics["silhouette"])
    logger.info("Calinski-Harabasz: %.3f",
                internal_metrics["calinski_harabasz"])
    logger.info("Davies-Bouldin: %.3f", internal_metrics["davies_bouldin"])

    metrics = {**external_metrics, **internal_metrics}

    output_dir = os.path.join(dataset_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    save_evaluation_results(
        dataset=accession,
        algorithm=algo_name,
        preprocessing=norm_method,
        with_pca=with_pca,
        metrics=metrics,
        output_dir=output_dir,
    )

    run_biological_evaluation(
        accession=accession,
        algo_name=algo_name,
        norm_method=norm_method,
        with_pca=with_pca,
        ground_truth=ground_truth,
        species=species,
        clustering_strategy=cluster_func,
        clustering_kwargs=cluster_kwargs,
        enrichment_set_names=enrichment_set_names,
    )


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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "data", accession)

    logger.info("Biological evaluation...")
    biological_input_filename = _processed_filename(
        norm_method, with_pca=False)
    biological_input_path = os.path.join(
        dataset_dir, "processed", biological_input_filename)
    if not os.path.exists(biological_input_path):
        raise FileNotFoundError(
            f"Biological input not found: {biological_input_filename}"
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


def _processed_filename(norm_method: NormMethod, with_pca: bool) -> str:
    return f"{norm_method}_{get_pca_label(with_pca)}.csv.gz"


def _hvg_filename(norm_method: NormMethod) -> str:
    return f"hvgs_{norm_method}.tsv"
