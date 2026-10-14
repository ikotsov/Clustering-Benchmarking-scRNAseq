import logging
from itertools import product

from src import paths
from src.constants import DATASETS, NORM_METHODS
from src.data_loading import load_dataset_config, parse_species
from src.logging_config import configure_logging
from src.runner import run_preprocessing


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
            config_path = paths.resolve(paths.DATASET_CONFIG, accession=dataset)
            species = parse_species(load_dataset_config(config_path))
            run_preprocessing(
                raw_path=paths.resolve(paths.RAW_COUNTS, accession=dataset),
                config_path=config_path,
                annotation_path=paths.resolve(paths.GENE_ANNOTATION, species=species),
                pca_output_path=paths.processed_data_path(dataset, norm_method, with_pca=True),
                no_pca_output_path=paths.processed_data_path(dataset, norm_method, with_pca=False),
                hvg_output_path=paths.resolve(paths.HVGS, accession=dataset, norm_method=norm_method),
                norm_method=norm_method,
            )
        except Exception:
            logger.exception("Failed preprocessing %s + %s", dataset, norm_method)
            continue
