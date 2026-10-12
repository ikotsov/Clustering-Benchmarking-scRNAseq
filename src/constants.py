from pathlib import Path

from src.types import NormMethod

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

SEED = 42
PCA_VARIANCE_RATIO = 0.80
RRNA_THRESHOLD = 0.05
APOPTOSIS_THRESHOLD = 0.05
MITO_THRESHOLD = 0.05
MIN_GENE_MAX_COUNT = 2
DATASETS = [
    "E-MTAB-3321",
    # "GSE36552",
    # "GSE45719",
    # "GSE57249",
]
NORM_METHODS: tuple[NormMethod, NormMethod] = ("log_cpm", "pearson")
