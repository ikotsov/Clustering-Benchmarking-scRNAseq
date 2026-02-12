import os
import pandas as pd
from src.data_loading import load_csv_data, load_dataset_config
from src.preprocessing import preprocess_data
from src.clustering.registry import get_clustering_strategy
from src.preprocessing.types import NormMethod
from src.evaluation import evaluate_clustering, save_evaluation_results
from src.constants import N_PCA_COMPONENTS


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
    species = config.get("species")

    # Load & preprocess
    print(
        f"\n=== PREPROCESSING: {accession}, norm_method={norm_method}, n_pca_components={n_pca_components} ===")
    print()
    raw_data = load_csv_data(raw_file_path)
    preprocessed_data = preprocess_data(
        raw_data,
        norm_method=norm_method,
        species=species if species else "human",
        n_pca_components=n_pca_components
    )

    # Save
    output_dir = os.path.join(dataset_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{norm_method}_pca_preprocessed.csv.gz"
    save_path = os.path.join(output_dir, filename)

    preprocessed_data.to_csv(save_path, compression='gzip')
    print()
    print(
        f"✓ Saved to: {filename} ({preprocessed_data.shape[0]} × {preprocessed_data.shape[1]} components)")


def run_experiment(accession: str, algo_name: str, norm_method: NormMethod = "pearson"):
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

    print(f"\n=== EXPERIMENT: {accession} + {algo_name.upper()} ===")
    print()

    # 2. Load preprocessed data
    preprocessed_filename = f"{norm_method}_pca_preprocessed.csv.gz"
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

    filename = f"{norm_method}_pca_{algo_name}.csv.gz"
    save_path = os.path.join(output_dir, filename)

    target_data.to_csv(save_path, compression='gzip')
    print()
    print(
        f"✓ Saved to: {filename} ({target_data.shape[0]} × {target_data.shape[1]} features + clusters)")

    # TODO: Get ground truth labels and provide to evaluation function
    # 5. Evaluate and save results
    # print()
    # print("Evaluation...")
    # labels_series = pd.Series(
    #     labels, index=target_data.index, name="cluster")
    # metrics = evaluate_clustering(labels_series, ground_truth)

    # print(f"  • ARI: {metrics['ari']:.3f}")
    # print(f"  • NMI: {metrics['nmi']:.3f}")

    # save_evaluation_results(
    #     dataset=accession,
    #     algorithm=algo_name,
    #     preprocessing=data_branch,
    #     n_pca_components=n_pca_components,
    #     metrics=metrics,
    #     output_dir=output_dir
    # )
