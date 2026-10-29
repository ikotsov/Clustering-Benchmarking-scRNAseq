"""
Prepare GSE60781 (Schlitzer et al., 2015; mouse bone marrow dendritic cell progenitors) as the standard counts + labels, e.g.:

    python scripts/prepare/GSE60781.py \\
        --cells data/datasets/GSE60781/raw/cells \\
        --series-matrix data/datasets/GSE60781/raw/GSE60781_series_matrix.txt.gz \\
        --out-counts data/datasets/GSE60781/interim/counts.csv.gz \\
        --out-labels data/datasets/GSE60781/interim/ground_truth_labels.csv

Each cell has its own HTSeq count file (gene, count), named after its GEO sample, e.g. GSM1488302_RMD072_readCount.txt.gz.
The labels are each sample's sorted population (CDP, MDP or PreDC), taken from its title in the GEO series
matrix (e.g. "CDP_1"). The HTSeq summary rows (e.g. "no_feature", "ambiguous") are dropped.
"""
import argparse
import csv
import gzip
from pathlib import Path

import pandas as pd

from src.prepare import save_prepared_dataset
from src.logging_config import configure_logging


HTSEQ_SUMMARY_ROWS = ["no_feature", "ambiguous", "too_low_aQual", "not_aligned", "alignment_not_unique"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True,
                        help="Directory of per-cell HTSeq count files (gzipped TSV: gene, count)")
    parser.add_argument("--series-matrix", type=Path, required=True,
                        help="GEO series matrix with the sample titles (gzipped)")
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    return parser.parse_args(argv)


def read_cell_counts(cells_dir: Path) -> pd.DataFrame:
    """Reads every per-cell count file into one cells x genes matrix, indexed by GEO sample accession."""
    cells = {
        path.name.split("_")[0]: pd.read_csv(path, sep="\t", header=None, index_col=0).iloc[:, 0]
        for path in sorted(cells_dir.glob("GSM*.txt.gz"))
    }
    counts = pd.DataFrame(cells).T
    return counts.drop(columns=HTSEQ_SUMMARY_ROWS)


def read_sample_populations(series_matrix: Path) -> pd.Series:
    """Reads each sample's population from its title in the series matrix (e.g. "CDP_1" -> "CDP")."""
    fields = {}
    with gzip.open(series_matrix, "rt") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0] in ("!Sample_geo_accession", "!Sample_title"):
                fields[row[0]] = row[1:]
    titles = pd.Series(fields["!Sample_title"], index=fields["!Sample_geo_accession"])
    return titles.str.rsplit("_", n=1).str[0]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    counts = read_cell_counts(args.cells)
    labels = read_sample_populations(args.series_matrix)

    save_prepared_dataset(counts, labels, args.out_counts, args.out_labels)


if __name__ == "__main__":
    main()
