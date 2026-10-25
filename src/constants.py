from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

SEED = 42
PCA_VARIANCE_RATIO = 0.80
MITO_THRESHOLD = 0.05
MIN_GENE_MAX_COUNT = 2
