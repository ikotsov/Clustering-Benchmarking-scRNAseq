"""
Collect the per-run result files of one dataset into its results table (TSV), e.g.:

    python scripts/aggregate_runs.py \\
        --dataset E-MTAB-3321 \\
        --out data/datasets/E-MTAB-3321/outputs/results.tsv \\
        data/datasets/E-MTAB-3321/outputs/runs/log_cpm/pca/kmeans.json \\
        data/datasets/E-MTAB-3321/outputs/runs/log_cpm/pca/leiden.json
"""
import argparse
from pathlib import Path

from src.evaluation import aggregate_run_results
from src.logging_config import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="Dataset accession ID (recorded in the table)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the results table (TSV)")
    parser.add_argument("run_results", type=Path, nargs="+",
                        help="Per-run result files (JSON), in the order of the table rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    aggregate_run_results(
        dataset=args.dataset,
        run_result_paths=args.run_results,
        results_path=args.out,
    )


if __name__ == "__main__":
    main()
