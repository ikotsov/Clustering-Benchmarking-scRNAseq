"""
Preprocess one dataset with one normalization method.

Filters and normalizes the raw counts, then saves the non-PCA and PCA representations
and the selected HVGs, e.g.:

    python scripts/preprocess.py \\
        --raw data/datasets/E-MTAB-3321/raw/E-MTAB-3321.csv.gz \\
        --config data/datasets/E-MTAB-3321/config.yaml \\
        --annotation data/annotations/gene_annotation_mouse.csv \\
        --norm-method log_cpm \\
        --out-pca data/datasets/E-MTAB-3321/processed/log_cpm_pca.csv.gz \\
        --out-no-pca data/datasets/E-MTAB-3321/processed/log_cpm_no_pca.csv.gz \\
        --out-hvgs data/datasets/E-MTAB-3321/processed/hvgs_log_cpm.tsv
"""
import argparse
from pathlib import Path
from typing import get_args

from src.constants import PCA_VARIANCE_RATIO
from src.logging_config import configure_logging
from src.runner import run_preprocessing
from src.types import NormMethod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Raw counts matrix (cells x genes, gzipped CSV)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Dataset config.yaml (species and QC thresholds)")
    parser.add_argument("--annotation", type=Path, required=True,
                        help="Gene annotation table of the dataset's species")
    parser.add_argument("--norm-method", choices=get_args(NormMethod), required=True)
    parser.add_argument("--pca-variance-ratio", type=float, default=PCA_VARIANCE_RATIO,
                        help="Fraction of variance kept by PCA (default: %(default)s)")
    parser.add_argument("--out-pca", type=Path, required=True)
    parser.add_argument("--out-no-pca", type=Path, required=True)
    parser.add_argument("--out-hvgs", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    run_preprocessing(
        raw_path=args.raw,
        config_path=args.config,
        annotation_path=args.annotation,
        pca_output_path=args.out_pca,
        no_pca_output_path=args.out_no_pca,
        hvg_output_path=args.out_hvgs,
        norm_method=args.norm_method,
        pca_variance_ratio=args.pca_variance_ratio,
    )


if __name__ == "__main__":
    main()
