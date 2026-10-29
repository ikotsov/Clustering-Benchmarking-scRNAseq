"""
Prepare E-MTAB-2805 (Buettner et al., 2015; mouse ESCs sorted by cell cycle phase) as the standard counts + labels, e.g.:

    python scripts/prepare/E-MTAB-2805.py \\
        --counts-g1 data/datasets/E-MTAB-2805/raw/G1_singlecells_counts.txt \\
        --counts-s data/datasets/E-MTAB-2805/raw/S_singlecells_counts.txt \\
        --counts-g2m data/datasets/E-MTAB-2805/raw/G2M_singlecells_counts.txt \\
        --out-counts data/datasets/E-MTAB-2805/interim/counts.csv.gz \\
        --out-labels data/datasets/E-MTAB-2805/interim/ground_truth_labels.csv

Each phase (sorted by Hoechst staining) has its own HTSeq count table (genes x cells), so the labels
are the phase of the file each cell comes from. ERCC spike-ins and the HTSeq summary rows
(e.g. "Aligned", "No_feature") are dropped, keeping only Ensembl genes.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.prepare import save_prepared_dataset
from src.logging_config import configure_logging


GENE_ID_COLUMN = "EnsemblGeneID"
# Gene annotation columns that precede the cell columns.
ANNOTATION_COLUMNS = ["EnsemblTranscriptID", "AssociatedGeneName", "GeneLength"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-g1", type=Path, required=True,
                        help="HTSeq count table of the G1 cells (genes x cells, TSV)")
    parser.add_argument("--counts-s", type=Path, required=True,
                        help="HTSeq count table of the S cells (genes x cells, TSV)")
    parser.add_argument("--counts-g2m", type=Path, required=True,
                        help="HTSeq count table of the G2M cells (genes x cells, TSV)")
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    return parser.parse_args(argv)


def read_phase_counts(path: Path) -> pd.DataFrame:
    """Reads one phase's count table as cells x genes, with cell IDs like "G1_cell1"."""
    table = pd.read_csv(path, sep="\t", index_col=GENE_ID_COLUMN).drop(columns=ANNOTATION_COLUMNS)
    genes = table[table.index.str.startswith("ENSMUSG")]
    return genes.T.rename(index=lambda cell: cell.removesuffix("_count"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    phase_counts = {
        "G1": read_phase_counts(args.counts_g1),
        "S": read_phase_counts(args.counts_s),
        "G2M": read_phase_counts(args.counts_g2m),
    }
    counts = pd.concat(phase_counts.values())
    labels = pd.concat(pd.Series(phase, index=c.index) for phase, c in phase_counts.items())

    save_prepared_dataset(counts, labels, args.out_counts, args.out_labels)


if __name__ == "__main__":
    main()
