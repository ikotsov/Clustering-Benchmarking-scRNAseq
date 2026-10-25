"""
Embed one PCA representation in 2D with UMAP and save the coordinates, for visualization only, e.g.:

    python scripts/embed.py \\
        --data data/datasets/E-MTAB-3321/processed/log_cpm_pca.csv.gz \\
        --out data/datasets/E-MTAB-3321/processed/umap_log_cpm.csv.gz
"""
import argparse
from pathlib import Path

from src.logging_config import configure_logging
from src.runner import run_umap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="PCA representation (gzipped CSV)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the UMAP coordinates (gzipped CSV)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    run_umap(
        pca_path=args.data,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
