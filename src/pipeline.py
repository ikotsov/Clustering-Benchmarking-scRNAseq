import os
from src.data_loading import load_csv_data
from src.preprocessing import preprocess_data
from src.clustering.registry import get_clustering_strategy


def run_experiment(accession: str, algo_name: str, data_branch: str = "pearson", **algo_params):
    """
    Orchestrates the full flow: Load -> Preprocess -> Cluster -> Save.
    """
    # 1. Dynamic path resolution
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "data", accession)

    raw_file_path = os.path.join(dataset_dir, "raw", f"{accession}.csv.gz")
    output_dir = os.path.join(dataset_dir, "results")

    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Expected file not found: {raw_file_path}")

    # 2. Load & preprocess
    print(f"--- Processing Dataset: {accession} ---")
    raw_data = load_csv_data(raw_file_path)
    target_data = preprocess_data(raw_data, branch=data_branch)

    # 3. Clustering
    cluster_func = get_clustering_strategy(algo_name)
    labels = cluster_func(target_data, **algo_params)

    # 4. Save
    os.makedirs(output_dir, exist_ok=True)

    final_df = target_data.copy()
    final_df["cluster"] = labels

    filename = f"{data_branch}_{algo_name}.csv.gz"
    save_path = os.path.join(output_dir, filename)

    final_df.to_csv(save_path, compression='gzip')
    print(f"Successfully saved {accession} results to: {save_path}")
