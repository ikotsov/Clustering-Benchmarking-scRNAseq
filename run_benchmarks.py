from itertools import product

from src.clustering.registry import AVAILABLE_ALGORITHMS, ClusteringAlgorithm
from src.constants import DATASETS
from src.preprocessing.types import NormMethod
from src.scripts import run_experiment


ALGORITHMS: tuple[ClusteringAlgorithm, ...] = tuple(AVAILABLE_ALGORITHMS)
NORM_METHODS: tuple[NormMethod, NormMethod] = ("log_cpm", "pearson")
PCA_OPTIONS: tuple[bool, bool] = (True, False)

if __name__ == "__main__":
    for algo_name, dataset, norm_method, with_pca in product(ALGORITHMS, DATASETS, NORM_METHODS, PCA_OPTIONS):
        run_experiment(
            accession=dataset,
            algo_name=algo_name,
            norm_method=norm_method,
            with_pca=with_pca,
        )
