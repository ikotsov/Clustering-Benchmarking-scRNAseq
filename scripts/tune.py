import logging
from itertools import product

from src import paths
from src.constants import DATASETS, NORM_METHODS
from src.logging_config import configure_logging
from src.tuning.algorithms import ALGORITHM_PARAM_SPECS, run_tuning
from src.tuning.common import aggregate_tuning_results


ALGORITHMS = tuple(ALGORITHM_PARAM_SPECS.keys())


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    configure_logging()
    total_runs = len(DATASETS) * len(ALGORITHMS) * len(NORM_METHODS)
    logger.info("Total runs: %s", total_runs)

    run_count = 0
    for dataset, algorithm, norm_method in product(DATASETS, ALGORITHMS, NORM_METHODS):
        run_count += 1
        logger.info(
            "[%s/%s] Tuning %s + %s + %s...",
            run_count,
            total_runs,
            dataset,
            algorithm,
            norm_method,
        )
        try:
            run_tuning(
                algorithm=algorithm,
                data_path=paths.processed_data_path(dataset, norm_method, with_pca=True),
                labels_path=paths.resolve(paths.GROUND_TRUTH_LABELS, accession=dataset),
                output_path=paths.resolve(
                    paths.TUNING_RESULT, accession=dataset, norm_method=norm_method, algorithm=algorithm),
                norm_method=norm_method,
            )
        except Exception:
            logger.exception(
                "Failed tuning %s + %s + %s", dataset, algorithm, norm_method)
            continue

    logger.info("Tuning complete! Results in:")
    for dataset in DATASETS:
        # Failed jobs leave no tuning result, so only the existing ones are collected.
        tuning_result_paths = [
            path
            for norm_method, algorithm in product(NORM_METHODS, ALGORITHMS)
            if (path := paths.resolve(
                paths.TUNING_RESULT, accession=dataset, norm_method=norm_method, algorithm=algorithm)).exists()
        ]
        clustering_params_path = paths.resolve(paths.CLUSTERING_PARAMS, accession=dataset)
        aggregate_tuning_results(tuning_result_paths, clustering_params_path)
        logger.info("%s", clustering_params_path)
