"""
Biological evaluation of one benchmark run: do the marker genes of its clusters match the reference sets
more than expression-matched random gene sets do? e.g.:

    python scripts/biological_evaluation.py \\
        --raw data/datasets/E-MTAB-3321/interim/counts.csv.gz \\
        --config data/datasets/E-MTAB-3321/config.yaml \\
        --annotation data/annotations/gene_annotation_mouse.csv \\
        --run-labels data/datasets/E-MTAB-3321/outputs/runs/log_cpm/pca/default/leiden_labels.csv \\
        --reference-sets data/datasets/E-MTAB-3321/outputs/reference_sets.json \\
        --n-samples 100 --min-marker-genes 10 \\
        --out data/datasets/E-MTAB-3321/outputs/runs/log_cpm/pca/default/leiden_biological.json
"""
import argparse
from pathlib import Path

from src.logging_config import configure_logging
from src.runner import run_biological_evaluation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Raw counts matrix, cells x genes (gzipped CSV)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Dataset config.yaml (QC thresholds, species)")
    parser.add_argument("--annotation", type=Path, required=True,
                        help="Gene annotation table of the dataset's species")
    parser.add_argument("--run-labels", type=Path, required=True,
                        help="Predicted cluster of each cell, saved by the benchmark run (CSV)")
    parser.add_argument("--reference-sets", type=Path, required=True,
                        help="Reference sets of the dataset (JSON, from scripts/reference_sets.py)")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="Random gene sets per cluster")
    parser.add_argument("--min-marker-genes", type=int, required=True,
                        help="Clusters with fewer marker genes are scored but not tested")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the evaluation (JSON)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    run_biological_evaluation(
        raw_path=args.raw,
        config_path=args.config,
        annotation_path=args.annotation,
        run_labels_path=args.run_labels,
        reference_sets_path=args.reference_sets,
        output_path=args.out,
        n_samples=args.n_samples,
        min_marker_genes=args.min_marker_genes,
    )


if __name__ == "__main__":
    main()
