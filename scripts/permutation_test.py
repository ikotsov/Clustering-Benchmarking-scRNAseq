"""
Build the null distribution of one benchmark run's metrics by shuffling its predicted labels, e.g.:

    python scripts/permutation_test.py \\
        --data data/datasets/E-MTAB-3321/processed/log_cpm/pca.csv.gz \\
        --run-labels data/datasets/E-MTAB-3321/outputs/runs/log_cpm/pca/tuned/leiden_labels.csv \\
        --labels data/datasets/E-MTAB-3321/interim/ground_truth_labels.csv \\
        --n-permutations 1000 \\
        --out data/datasets/E-MTAB-3321/outputs/runs/log_cpm/pca/tuned/leiden_null.npz
"""
import argparse
from pathlib import Path

from src.logging_config import configure_logging
from src.runner import run_permutation_test


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="Preprocessed data the run clustered (gzipped CSV)")
    parser.add_argument("--run-labels", type=Path, required=True,
                        help="Predicted cluster of each cell, saved by the benchmark run (CSV)")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Ground truth labels CSV")
    parser.add_argument("--n-permutations", type=int, required=True,
                        help="Number of shuffles of the predicted labels")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the null distribution (.npz)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    run_permutation_test(
        data_path=args.data,
        run_labels_path=args.run_labels,
        labels_path=args.labels,
        n_permutations=args.n_permutations,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
