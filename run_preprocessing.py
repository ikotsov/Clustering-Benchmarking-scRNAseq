import logging
from itertools import product

from src.constants import DATASETS, NORM_METHODS
from src.logging_config import configure_logging
from src.scripts import run_preprocessing


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_logging()
    total_runs = len(DATASETS) * len(NORM_METHODS)
    logger.info("Total runs: %s", total_runs)

    run_count = 0
    for dataset, norm_method in product(DATASETS, NORM_METHODS):
        run_count += 1
        logger.info("[%s/%s] Preprocessing %s + %s...",
                    run_count, total_runs, dataset, norm_method)
        try:
            run_preprocessing(
                accession=dataset,
                norm_method=norm_method,
            )
        except Exception as e:
            logger.error("Error: %s", e)
            continue
