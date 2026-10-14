import logging
from itertools import product

from src import paths
from src.clustering.registry import AVAILABLE_ALGORITHMS, ClusteringAlgorithm
from src.constants import DATASETS, NORM_METHODS
from src.evaluation import aggregate_run_results
from src.logging_config import configure_logging
from src.runner import run_experiment
from src.tuning.algorithms import ALGORITHM_PARAM_SPECS


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
            data_path=paths.processed_data_path(dataset, norm_method, with_pca),
            config_path=paths.resolve(paths.DATASET_CONFIG, accession=dataset),
            # Untuned algorithms run with their defaults.
            params_path=paths.resolve(
                paths.TUNING_RESULT, accession=dataset, norm_method=norm_method, algorithm=algo_name)
            if algo_name in ALGORITHM_PARAM_SPECS else None,
            labels_path=paths.resolve(paths.GROUND_TRUTH_LABELS, accession=dataset),
            result_path=paths.run_result_path(dataset, norm_method, with_pca, algo_name),
            norm_method=norm_method,
            with_pca=with_pca,
        )

    for dataset in DATASETS:
        aggregate_run_results(
            dataset=dataset,
            run_result_paths=[
                paths.run_result_path(dataset, norm_method, with_pca, algo_name)
                for algo_name, norm_method, with_pca in product(ALGORITHMS, NORM_METHODS, PCA_OPTIONS)
            ],
            results_path=paths.resolve(paths.RESULTS, accession=dataset),
        )
