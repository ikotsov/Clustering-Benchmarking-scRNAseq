import os
import pandas as pd
from src.data_loading import load_csv_data, load_dataset_config, load_ground_truth_labels, parse_preprocessing_config
from src.preprocessing import preprocess_data
from src.clustering.registry import get_clustering_strategy
from src.preprocessing.types import NormMethod
from src.evaluation import evaluate_clustering, save_evaluation_results
from src.constants import N_PCA_COMPONENTS
from src.types import Species


def run_preprocessing(accession: str, norm_method: NormMethod = "pearson", n_pca_components: int = N_PCA_COMPONENTS):
    """
    Runs only the preprocessing step and saves the result.
    Useful for exploring preprocessing outputs or running multiple preprocessing strategies for all datasets at once.

    Parameters
    ----------
    accession : str
        Dataset accession ID
    norm_method : NormMethod, default="pearson"
        Normalization method ("pearson" or "log_cpm")
    n_pca_components : int, default=N_PCA_COMPONENTS
        Number of PCA components to retain
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "data", accession)

    raw_file_path = os.path.join(dataset_dir, "raw", f"{accession}.csv.gz")

    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Expected file not found: {raw_file_path}")

    config = load_dataset_config(dataset_dir)
    species_value = config.get("species")
    species = species_value if species_value in ["human", "mouse"] else "human"
    preprocessing_config = parse_preprocessing_config(config)

    # Load & preprocess
    print(
        f"\n=== PREPROCESSING: {accession}, norm_method={norm_method}, n_pca_components={n_pca_components} ===")
    print()
    raw_data = load_csv_data(raw_file_path)
    preprocessed_pca = preprocess_data(
        raw_data,
        norm_method=norm_method,
        species=species,
        n_pca_components=n_pca_components,
        preprocessing_config=preprocessing_config,
        with_pca=True,
    )
    preprocessed_no_pca = preprocess_data(
        raw_data,
        norm_method=norm_method,
        species=species,
        n_pca_components=n_pca_components,
        preprocessing_config=preprocessing_config,
        with_pca=False,
    )

    # Save both representations
    output_dir = os.path.join(dataset_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    for with_pca in (True, False):
        preprocessed_data = preprocessed_pca if with_pca else preprocessed_no_pca
        filename = _preprocessed_filename(norm_method, with_pca)
        save_path = os.path.join(output_dir, filename)
        preprocessed_data.to_csv(save_path, compression='gzip')
        print()
        print(
            f"✓ Saved to: {filename} ({preprocessed_data.shape[0]} × {preprocessed_data.shape[1]} features)")


def run_experiment(
    accession: str,
    algo_name: str,
    norm_method: NormMethod = "pearson",
    with_pca: bool = True,
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

    pca_tag = _pca_label(with_pca).upper()
    print(
        f"\n=== EXPERIMENT: {accession} + {algo_name.upper()} + {pca_tag} ===")
    print()

    # 2. Load preprocessed data
    preprocessed_filename = _preprocessed_filename(norm_method, with_pca)
    preprocessed_file = os.path.join(
        dataset_dir, "results", preprocessed_filename)

    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(
            f"Preprocessed file not found: {preprocessed_filename}\n"
            f"Run preprocessing first with: run_preprocessing('{accession}', '{norm_method}')"
        )

    print(f"Loading preprocessed data: {preprocessed_filename}")
    target_data = load_csv_data(preprocessed_file)

    # 3. Clustering
    config = load_dataset_config(dataset_dir)
    print()
    print(f"Clustering ({algo_name})...")
    cluster_func = get_clustering_strategy(algo_name)

    n_clusters = config.get("n_clusters")
    labels = cluster_func(
        target_data, n_clusters=n_clusters if n_clusters else 5)

    # 4. Save results
    output_dir = os.path.join(dataset_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    target_data["cluster"] = labels

    filename = f"{norm_method}_{_pca_label(with_pca)}_{algo_name}.csv.gz"
    save_path = os.path.join(output_dir, filename)

    target_data.to_csv(save_path, compression='gzip')
    print()
    print(
        f"✓ Saved to: {filename} ({target_data.shape[0]} × {target_data.shape[1]} features + clusters)")

    # 5. Load ground truth labels
    print()
    print("Loading ground truth labels...")
    try:
        ground_truth = load_ground_truth_labels(dataset_dir)
        print(f"  ✓ Loaded {len(ground_truth)} ground truth labels")
    except FileNotFoundError as e:
        print(f"  ⚠ Warning: {e}")
        print("  Skipping evaluation.")
        return

    # 6. Evaluate and save results
    print()
    print("Evaluation...")
    # Here we align the predicted labels with the ground truth labels based on the index (cell IDs).
    # The alignment is done based on the order of the indices in the target_data.
    # Wew rely on the fact that the clustering algorithm returns labels in the same order as the input data.
    labels_series = pd.Series(
        labels, index=target_data.index, name="cluster")
    metrics = evaluate_clustering(labels_series, ground_truth)

    print(f"  • ARI: {metrics['ari']:.3f}")
    print(f"  • NMI: {metrics['nmi']:.3f}")

    save_evaluation_results(
        dataset=accession,
        algorithm=algo_name,
        preprocessing=f"{norm_method}_{_pca_label(with_pca)}",
        n_pca_components=N_PCA_COMPONENTS if with_pca else 0,
        metrics=metrics,
        output_dir=output_dir
    )


def _pca_label(with_pca: bool) -> str:
    return "pca" if with_pca else "no_pca"


def _preprocessed_filename(norm_method: NormMethod, with_pca: bool) -> str:
    return f"{norm_method}_{_pca_label(with_pca)}_preprocessed.csv.gz"
