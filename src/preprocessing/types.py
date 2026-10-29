from dataclasses import dataclass
from src.constants import MIN_GENE_MAX_COUNT, MITO_THRESHOLD


@dataclass
class PreprocessingConfig:
    mito_threshold: float = MITO_THRESHOLD
    min_gene_max_count: int = MIN_GENE_MAX_COUNT
    count_threshold: float | None = None
    gene_threshold: int | None = None
