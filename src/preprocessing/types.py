from dataclasses import dataclass
from typing import Literal

from src.constants import GENE_MAGNITUDE_THRESHOLD

Species = Literal["human", "mouse"]
NormMethod = Literal["pearson", "log_cpm"]


@dataclass
class PreprocessingConfig:
    mito_threshold: float = 0.05
    rrna_threshold: float = 0.05
    apoptosis_threshold: float = 0.05
    gene_magnitude_threshold: int = GENE_MAGNITUDE_THRESHOLD
