from dataclasses import dataclass
from src.constants import MIN_GENE_MAX_COUNT, RRNA_THRESHOLD, APOPTOSIS_THRESHOLD, MITO_THRESHOLD


@dataclass
class PreprocessingConfig:
    mito_threshold: float = MITO_THRESHOLD
    rrna_threshold: float = RRNA_THRESHOLD
    apoptosis_threshold: float = APOPTOSIS_THRESHOLD
    min_gene_max_count: int = MIN_GENE_MAX_COUNT
