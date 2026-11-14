"""
Prepare the CellBench CEL-seq2 experiment with 5 human lung adenocarcinoma cell lines on 3 plates (Tian et al.,
2019; GSE118767, samples GSM3618022-24) as one dataset of standard counts + labels, e.g.:

    python scripts/prepare/GSE118767_celseq2_5cl.py \\
        --counts-p1 data/datasets/GSE118767_celseq2_5cl/raw/GSM3618022_gene_count_RPI4.csv.gz \\
        --counts-p2 data/datasets/GSE118767_celseq2_5cl/raw/GSM3618023_gene_count_RPI5.csv.gz \\
        --counts-p3 data/datasets/GSE118767_celseq2_5cl/raw/GSM3618024_gene_count_RPI6.csv.gz \\
        --metadata-p1 data/datasets/GSE118767_celseq2_5cl/raw/sc_celseq2_5cl_p1.metadata.csv.gz \\
        --metadata-p2 data/datasets/GSE118767_celseq2_5cl/raw/sc_celseq2_5cl_p2.metadata.csv.gz \\
        --metadata-p3 data/datasets/GSE118767_celseq2_5cl/raw/sc_celseq2_5cl_p3.metadata.csv.gz \\
        --out-counts data/datasets/GSE118767_celseq2_5cl/interim/counts.csv.gz \\
        --out-labels data/datasets/GSE118767_celseq2_5cl/interim/ground_truth_labels.csv

Each count matrix is genes x the wells of one 384-well plate (RPI4-6 are plates p1-p3). GEO has no per-cell
labels, so they come from the authors' metadata
(https://github.com/LuyiTian/sc_mixology/tree/master/data/csv, sc_celseq2_5cl_p*.metadata.csv.gz), see
read_cellbench_labels. Only the labeled wells are kept, and the ERCC spike-ins are dropped.

The metadata of every plate names its wells "p1_<well>", so the prefix is replaced by the plate's own, which also
keeps the cell IDs unique across plates. The plates are merged into one dataset: their cells group by cell line,
not by plate. A gene missing from a plate's matrix had no counts on that plate, so it is filled with 0.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.prepare import read_cellbench_labels, read_genes_by_cells_csv, save_prepared_dataset
from src.logging_config import configure_logging

PLATES = ("p1", "p2", "p3")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for plate in PLATES:
        parser.add_argument(f"--counts-{plate}", type=Path, required=True,
                            help=f"Count matrix of plate {plate} (genes x wells, gzipped CSV)")
        parser.add_argument(f"--metadata-{plate}", type=Path, required=True,
                            help=f"CellBench cell metadata of plate {plate} (gzipped CSV, one row per QC-passed well)")
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    return parser.parse_args(argv)


def read_plate(counts_path: Path, metadata_path: Path, plate: str) -> tuple[pd.DataFrame, pd.Series]:
    labels = read_cellbench_labels(metadata_path)
    wells = labels.index.str.replace(r"^p\d+_", "", regex=True)
    counts = read_genes_by_cells_csv(counts_path).loc[wells]
    counts = counts.loc[:, ~counts.columns.str.startswith("ERCC-")]
    counts.index = f"{plate}_" + counts.index
    labels.index = f"{plate}_" + wells
    return counts, labels


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    plates = [read_plate(getattr(args, f"counts_{p}"), getattr(args, f"metadata_{p}"), p) for p in PLATES]
    counts = pd.concat([c for c, _ in plates]).fillna(0).astype(int)
    labels = pd.concat([l for _, l in plates])

    save_prepared_dataset(counts, labels, args.out_counts, args.out_labels)


if __name__ == "__main__":
    main()
