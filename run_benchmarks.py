from src.scripts import run_experiment

if __name__ == "__main__":
    # Experiment 1: Standard K-Means on log-CPM data
    run_experiment(accession="E-MTAB-3321",
                   algo_name="kmeans", norm_method="log_cpm")
