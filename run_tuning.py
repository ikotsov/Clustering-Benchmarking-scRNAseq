import logging
from itertools import product

from src.constants import DATASETS, NORM_METHODS
from src.logging_config import configure_logging
from src.tuning.algorithms import ALGORITHM_PARAM_SPECS, run_tuning
from src.tuning.common import CLUSTERING_PARAMS_FILENAME


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
                accession=dataset,
                algorithm=algorithm,
                norm_method=norm_method,
            )
        except Exception as e:
            logger.error("Error: %s", e)
            continue

    logger.info("Tuning complete! Results in:")
    for dataset in DATASETS:
        logger.info("data/%s/outputs/%s", dataset, CLUSTERING_PARAMS_FILENAME)
