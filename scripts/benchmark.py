"""
Run one benchmark: cluster one preprocessed dataset with one algorithm, evaluate it and save the metrics, e.g.:

    python scripts/benchmark.py \\
        --dataset E-MTAB-3321 \\
        --algorithm leiden \\
        --norm-method log_cpm \\
        --pca \\
        --data data/datasets/E-MTAB-3321/processed/log_cpm_pca.csv.gz \\
        --config data/datasets/E-MTAB-3321/config.yaml \\
        --labels data/datasets/E-MTAB-3321/interim/ground_truth_labels.csv \\
        --params data/datasets/E-MTAB-3321/outputs/tuning/log_cpm/pca/leiden.json \\
        --out data/datasets/E-MTAB-3321/outputs/runs/log_cpm/pca/tuned/leiden.json

Leave out --params to run the algorithm with its defaults.
"""
import argparse
from pathlib import Path
from typing import get_args

from src.clustering.registry import AVAILABLE_ALGORITHMS
from src.logging_config import configure_logging
from src.runner import run_experiment
from src.types import NormMethod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="Dataset accession ID (recorded in the results)")
    parser.add_argument("--algorithm", choices=AVAILABLE_ALGORITHMS, required=True)
    parser.add_argument("--norm-method", choices=get_args(NormMethod), required=True,
                        help="Normalization the data was preprocessed with (recorded in the result)")
    parser.add_argument("--pca", action=argparse.BooleanOptionalAction, required=True,
                        help="Whether --data is the PCA representation (--pca / --no-pca)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Preprocessed data to cluster (gzipped CSV)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Dataset config.yaml (species and n_clusters)")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Ground truth labels CSV")
    parser.add_argument("--params", type=Path, default=None,
                        help="Tuning result of this algorithm + normalization + PCA option (JSON)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the run's metrics (JSON)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    run_experiment(
        accession=args.dataset,
        algo_name=args.algorithm,
        data_path=args.data,
        config_path=args.config,
        params_path=args.params,
        labels_path=args.labels,
        result_path=args.out,
        norm_method=args.norm_method,
        with_pca=args.pca,
    )


if __name__ == "__main__":
    main()
