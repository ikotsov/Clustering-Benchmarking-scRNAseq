"""
Collect the per-job tuning result files of one dataset into its clustering_params.json, e.g.:

    python scripts/aggregate_tuning.py \\
        --out data/datasets/E-MTAB-3321/outputs/clustering_params.json \\
        data/datasets/E-MTAB-3321/outputs/tuning/log_cpm/leiden.json \\
        data/datasets/E-MTAB-3321/outputs/tuning/log_cpm/birch.json
"""
import argparse
from pathlib import Path

from src.logging_config import configure_logging
from src.tuning.common import aggregate_tuning_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the collected tuning results (JSON)")
    parser.add_argument("tuning_results", type=Path, nargs="+",
                        help="Per-job tuning result files (JSON)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    aggregate_tuning_results(
        tuning_result_paths=args.tuning_results,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
