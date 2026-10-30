"""
Prepare zheng_pbmc_bead_enriched (Zheng et al., 2017; bead-enriched human PBMC populations, sampled in the lab)
as the standard counts + labels, e.g.:

    python scripts/prepare/zheng_pbmc_bead_enriched.py \\
        --counts data/datasets/zheng_pbmc_bead_enriched/raw/pbmc.bead.enriched.sample.zheng.count_matrix.csv.gz \\
        --metadata data/datasets/zheng_pbmc_bead_enriched/raw/pbmc.bead.enriched.sample.zheng.metadata.bminty.txt.gz \\
        --out-counts data/datasets/zheng_pbmc_bead_enriched/interim/counts.csv.gz \\
        --out-labels data/datasets/zheng_pbmc_bead_enriched/interim/ground_truth_labels.csv

The count matrix is genes x cells; the labels are the population each cell was purified from
("barcode_cell_type|cell_type" metadata).
"""
import argparse
from pathlib import Path

from src.prepare import read_genes_by_cells_csv, read_pipe_separated_labels, save_prepared_dataset
from src.logging_config import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=Path, required=True,
                        help="Source count matrix (genes x cells, gzipped CSV)")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Source metadata (pipe-separated: barcode_cell_type|cell_type)")
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    counts = read_genes_by_cells_csv(args.counts)
    labels = read_pipe_separated_labels(args.metadata, cell_column="barcode_cell_type", label_column="cell_type")
    save_prepared_dataset(counts, labels, args.out_counts, args.out_labels)


if __name__ == "__main__":
    main()
