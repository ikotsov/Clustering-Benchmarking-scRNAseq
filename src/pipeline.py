import os
from src.data_loading import load_10x_data
from src.preprocessing import preprocess_data
from src.clustering.registry import get_clustering_strategy


def run_experiment(algo_name: str, data_branch: str = "pearson", **algo_params):
    """
    Orchestrates the full flow: Load -> Preprocess -> Cluster -> Save.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "10x")
    output_dir = os.path.join(project_root, "data")

    raw_data = load_10x_data(data_path)
    # preprocess_data returns: {"pearson": df, "log_cpm": df}
    processed_dict = preprocess_data(raw_data)

    if data_branch not in processed_dict:
        raise ValueError(
            f"Branch {data_branch} not found. Options: {list(processed_dict.keys())}")

    target_data = processed_dict[data_branch]

    cluster_func = get_clustering_strategy(algo_name)
    labels = cluster_func(target_data, **algo_params)

    final_df = target_data.copy()
    final_df["cluster"] = labels

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"results_{algo_name}_{data_branch}.csv"
    final_df.to_csv(os.path.join(output_dir, file_name))

    print(f"--- Experiment Finished ---")
    print(
        f"Algorithm: {algo_name} | Branch: {data_branch} | Saved: {file_name}")
