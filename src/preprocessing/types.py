from dataclasses import dataclass
from src.constants import GENE_MAGNITUDE_THRESHOLD, RRNA_THRESHOLD, APOPTOSIS_THRESHOLD, MITO_THRESHOLD


@dataclass
class PreprocessingConfig:
    mito_threshold: float = MITO_THRESHOLD
    rrna_threshold: float = RRNA_THRESHOLD
    apoptosis_threshold: float = APOPTOSIS_THRESHOLD
    gene_magnitude_threshold: int = GENE_MAGNITUDE_THRESHOLD
