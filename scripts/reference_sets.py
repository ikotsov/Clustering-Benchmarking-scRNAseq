"""
Build the reference sets of one dataset for the biological evaluation: the DE markers of each ground-truth
cell type, plus the marker databases set in the dataset's config.yaml, e.g.:

    python scripts/reference_sets.py \\
        --raw data/datasets/zheng_pbmc_bead_enriched/interim/counts.csv.gz \\
        --config data/datasets/zheng_pbmc_bead_enriched/config.yaml \\
        --annotation data/annotations/gene_annotation_human.csv \\
        --labels data/datasets/zheng_pbmc_bead_enriched/interim/ground_truth_labels.csv \\
        --marker-index data/gene_markers/outputs/gene_marker_index.json \\
        --out data/datasets/zheng_pbmc_bead_enriched/outputs/reference_sets.json
"""
import argparse
from pathlib import Path

from src.logging_config import configure_logging
from src.runner import run_reference_sets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Raw counts matrix, cells x genes (gzipped CSV)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Dataset config.yaml (QC thresholds, species, marker databases)")
    parser.add_argument("--annotation", type=Path, required=True,
                        help="Gene annotation table of the dataset's species")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Ground truth labels CSV")
    parser.add_argument("--marker-index", type=Path, default=None,
                        help="Gene marker index (JSON); needed when the config sets marker databases")
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to save the reference sets (JSON)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    run_reference_sets(
        raw_path=args.raw,
        config_path=args.config,
        annotation_path=args.annotation,
        labels_path=args.labels,
        marker_index_path=args.marker_index,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
