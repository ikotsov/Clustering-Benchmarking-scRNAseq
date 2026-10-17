"""
Prepare E-MTAB-3321 (Goolam et al., 2015; mouse preimplantation embryos) as the standard counts + labels, e.g.:

    python scripts/prepare/E-MTAB-3321.py \\
        --counts data/datasets/E-MTAB-3321/raw/Goolam_et_al_2015_count_table.tsv \\
        --metadata data/datasets/E-MTAB-3321/raw/E-MTAB-3321.sdrf.txt \\
        --out-counts data/datasets/E-MTAB-3321/interim/counts.csv.gz \\
        --out-labels data/datasets/E-MTAB-3321/interim/ground_truth_labels.csv

The count table is genes x cells; the labels are each cell's developmental stage from the SDRF metadata.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.prepare import save_prepared_dataset
from src.logging_config import configure_logging


CELL_ID_COLUMN = "Factor Value[single cell identifier]"
LABEL_COLUMN = "Characteristics[developmental stage]"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=Path, required=True,
                        help="Source count table (genes x cells, TSV)")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Source SDRF metadata (TSV)")
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    # The header has one field fewer than the rows, so the gene IDs become the index.
    counts = pd.read_csv(args.counts, sep="\t").T

    metadata = pd.read_csv(args.metadata, sep="\t")
    # The metadata has one row per technical replicate, so each cell appears more than once.
    labels = metadata.drop_duplicates(subset=[CELL_ID_COLUMN]).set_index(CELL_ID_COLUMN)[LABEL_COLUMN]

    save_prepared_dataset(counts, labels, args.out_counts, args.out_labels)


if __name__ == "__main__":
    main()
