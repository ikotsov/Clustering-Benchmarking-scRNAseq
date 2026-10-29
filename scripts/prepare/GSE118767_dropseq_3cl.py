"""
Prepare the CellBench Drop-seq experiment with 3 human lung adenocarcinoma cell lines (Tian et al., 2019; GSE118767,
sample GSM3336849) as the standard counts + labels, e.g.:

    python scripts/prepare/GSE118767_dropseq_3cl.py \\
        --counts data/datasets/GSE118767_dropseq_3cl/raw/GSM3336849_dropseq_gene_count.csv.gz \\
        --metadata data/datasets/GSE118767_dropseq_3cl/raw/sc_dropseq.metadata.csv.gz \\
        --out-counts data/datasets/GSE118767_dropseq_3cl/interim/counts.csv.gz \\
        --out-labels data/datasets/GSE118767_dropseq_3cl/interim/ground_truth_labels.csv

The count matrix is genes x barcodes. GEO has no per-cell labels, so they come from the authors' metadata
(https://github.com/LuyiTian/sc_mixology/blob/master/data/csv/sc_dropseq.metadata.csv.gz), see read_cellbench_labels.
Only the labeled barcodes are kept; the others are mostly empty droplets.
"""
import argparse
from pathlib import Path

from src.prepare import read_cellbench_labels, read_genes_by_cells_csv, save_prepared_dataset
from src.logging_config import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=Path, required=True,
                        help="Source count matrix (genes x barcodes, gzipped CSV)")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="CellBench cell metadata (gzipped CSV, one row per QC-passed barcode)")
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    labels = read_cellbench_labels(args.metadata)
    counts = read_genes_by_cells_csv(args.counts).loc[labels.index]

    save_prepared_dataset(counts, labels, args.out_counts, args.out_labels)


if __name__ == "__main__":
    main()
