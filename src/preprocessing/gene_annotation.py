import pandas as pd

from ..paths import GENE_ANNOTATION, resolve
from ..types import PathLike, Species


# Status columns hold the gene name when the gene belongs to the set, and a "NO_..." placeholder otherwise.
MITO_STATUS_COLUMN = "MT_status"
RRNA_STATUS_COLUMN = "RB_status"
APOPTOSIS_STATUS_COLUMN = "APOPT_status"


def load_gene_annotation(species: Species = "human", annotation_path: PathLike | None = None) -> pd.DataFrame:
    """
    Loads the Ensembl gene annotation table of a species
    (gene_id, gene_name, gene_biotype, chromosome, gene set status columns, alias_names).
    Reads annotation_path when given, otherwise the project's annotation file for the species.
    """
    if annotation_path is None:
        annotation_path = resolve(GENE_ANNOTATION, species=species)
    return pd.read_csv(annotation_path)
