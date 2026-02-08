# run_benchmarks.py
from src.pipeline import run_experiment

if __name__ == "__main__":
    # Experiment 1: Standard K-Means on Pearson
    run_experiment(accession="E-MTAB-3321", algo_name="kmeans",
                   data_branch="pearson", n_clusters=5)

    # Experiment 2: Leiden on LogCPM to see the difference
    # run_experiment(accession="E-MTAB-3321", algo_name="leiden", data_branch="log_cpm", resolution=0.8)

    # Experiment 3: Spectral Clustering on Pearson
    # run_experiment(accession="E-MTAB-3321", algo_name="spectral", data_branch="pearson", n_clusters=5)
