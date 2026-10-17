"""
Prepare GSE36552 (Yan et al., 2013; human preimplantation embryos and hESCs) as the standard counts + labels, e.g.:

    python scripts/prepare/GSE36552.py \\
        --counts data/datasets/GSE36552/raw/GSE36552.csv.gz \\
        --metadata data/datasets/GSE36552/raw/GSE36552.metadata.bminty.txt.gz \\
        --out-counts data/datasets/GSE36552/interim/counts.csv.gz \\
        --out-labels data/datasets/GSE36552/interim/ground_truth_labels.csv

The count matrix is genes x cells; the labels are each cell's type from the "Cell|Type" metadata.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.prepare import save_prepared_dataset
from src.logging_config import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=Path, required=True,
                        help="Source count matrix (genes x cells, gzipped CSV)")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Source metadata (pipe-separated: Cell|Type)")
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    counts = pd.read_csv(args.counts, index_col=0).T

    metadata = pd.read_csv(args.metadata, sep="|")
    # Names and values carry stray whitespace (e.g. "Oocyte ").
    metadata.columns = metadata.columns.str.strip()
    labels = metadata.set_index(metadata["Cell"].str.strip())["Type"].str.strip()

    save_prepared_dataset(counts, labels, args.out_counts, args.out_labels)


if __name__ == "__main__":
    main()
