import os
from src.data_loading import load_csv_data, load_dataset_config
from src.preprocessing import preprocess_data
from src.clustering.registry import get_clustering_strategy


def run_experiment(accession: str, algo_name: str, data_branch: str = "pearson"):
    """
    Orchestrates the full flow: Load -> Preprocess -> Cluster -> Save.
    """
    # 1. Dynamic path resolution
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "data", accession)

    raw_file_path = os.path.join(dataset_dir, "raw", f"{accession}.csv.gz")

    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Expected file not found: {raw_file_path}")

    config = load_dataset_config(dataset_dir)
    species = config.get("species")

    # 2. Load & preprocess
    print(f"--- Processing Dataset: {accession} ---")
    raw_data = load_csv_data(raw_file_path)
    target_data = preprocess_data(
        raw_data, branch=data_branch, species=species if species else "human")

    # 3. Clustering
    cluster_func = get_clustering_strategy(algo_name)

    n_clusters = config.get("n_clusters")
    labels = cluster_func(
        target_data, n_clusters=n_clusters if n_clusters else 5)

    # 4. Save
    output_dir = os.path.join(dataset_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    target_data["cluster"] = labels

    filename = f"{data_branch}_{algo_name}.csv.gz"
    save_path = os.path.join(output_dir, filename)

    target_data.to_csv(save_path, compression='gzip')
    print(f"Successfully saved {accession} results to: {save_path}")
