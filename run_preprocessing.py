from src.pipeline import run_preprocessing


if __name__ == "__main__":
    run_preprocessing(accession="E-MTAB-3321", data_branch="pearson")
    run_preprocessing(accession="E-MTAB-3321", data_branch="log_cpm")
