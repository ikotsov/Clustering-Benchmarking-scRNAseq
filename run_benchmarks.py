from src.scripts import run_experiment

if __name__ == "__main__":
    # Experiment 1: Standard K-Means on Pearson
    run_experiment(accession="E-MTAB-3321",
                   algo_name="kmeans", norm_method="pearson")
