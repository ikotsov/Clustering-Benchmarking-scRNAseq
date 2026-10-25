"""
Tune the parameters of one clustering algorithm on one preprocessed dataset.

Runs an Optuna search on the given representation (PCA or not) and saves the best parameters, e.g.:

    python scripts/tune.py \\
        --data data/datasets/E-MTAB-3321/processed/log_cpm/pca.csv.gz \\
        --config data/datasets/E-MTAB-3321/config.yaml \\
        --labels data/datasets/E-MTAB-3321/interim/ground_truth_labels.csv \\
        --algorithm leiden \\
        --norm-method log_cpm \\
        --pca \\
        --out data/datasets/E-MTAB-3321/outputs/tuning/log_cpm/pca/leiden.json
"""
import argparse
from pathlib import Path
from typing import get_args

from src.logging_config import configure_logging
from src.tuning.algorithms import ALGORITHM_PARAM_SPECS, run_tuning
from src.tuning.common import ObjectiveMetric
from src.types import NormMethod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="Preprocessed data to tune on (gzipped CSV)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Dataset config.yaml (n_clusters)")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Ground truth labels CSV")
    parser.add_argument("--algorithm", choices=tuple(ALGORITHM_PARAM_SPECS), required=True)
    parser.add_argument("--norm-method", choices=get_args(NormMethod), required=True,
                        help="Normalization the data was preprocessed with (recorded in the result)")
    parser.add_argument("--pca", action=argparse.BooleanOptionalAction, required=True,
                        help="Whether --data is the PCA representation (--pca / --no-pca)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: %(default)s)")
    parser.add_argument("--objective-metric", choices=get_args(ObjectiveMetric), default="mean_external_score",
                        help="Metric to maximize (default: %(default)s)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the tuning result (JSON)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    run_tuning(
        algorithm=args.algorithm,
        data_path=args.data,
        config_path=args.config,
        labels_path=args.labels,
        output_path=args.out,
        n_trials=args.n_trials,
        norm_method=args.norm_method,
        with_pca=args.pca,
        objective_metric=args.objective_metric,
    )


if __name__ == "__main__":
    main()
