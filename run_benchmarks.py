import logging
from itertools import product

from src.clustering.registry import AVAILABLE_ALGORITHMS, ClusteringAlgorithm
from src.constants import DATASETS, NORM_METHODS
from src.logging_config import configure_logging
from src.scripts import run_experiment


ALGORITHMS: tuple[ClusteringAlgorithm, ...] = tuple(AVAILABLE_ALGORITHMS)
PCA_OPTIONS: tuple[bool, bool] = (True, False)


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_logging()
    for algo_name, dataset, norm_method, with_pca in product(ALGORITHMS, DATASETS, NORM_METHODS, PCA_OPTIONS):
        logger.info(
            "Benchmarking %s + %s + %s + %s",
            dataset,
            algo_name,
            norm_method,
            "pca" if with_pca else "no_pca",
        )
        run_experiment(
            accession=dataset,
            algo_name=algo_name,
            norm_method=norm_method,
            with_pca=with_pca,
        )
