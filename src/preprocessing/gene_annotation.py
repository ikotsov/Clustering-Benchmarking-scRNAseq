
import pandas as pd

from ..constants import DATA_DIR
from ..types import Species


ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Status columns hold the gene name when the gene belongs to the set, and a "NO_..." placeholder otherwise.
MITO_STATUS_COLUMN = "MT_status"


def load_gene_annotation(species: Species = "human") -> pd.DataFrame:
    """
    Loads the Ensembl gene annotation table of a species
    (gene_id, gene_name, gene_biotype, chromosome, gene set status columns, alias_names).
    """
    return pd.read_csv(ANNOTATIONS_DIR / f"gene_annotation_{species}.csv")
