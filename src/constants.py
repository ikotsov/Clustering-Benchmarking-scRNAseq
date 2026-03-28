from src.types import NormMethod

SEED = 42
PCA_VARIANCE_RATIO = 0.80
RRNA_THRESHOLD = 0.05
APOPTOSIS_THRESHOLD = 0.05
MITO_THRESHOLD = 0.05
GENE_MAGNITUDE_THRESHOLD = 2
DATASETS = [
    "E-MTAB-3321",
    # "GSE36552",
    # "GSE45719",
    # "GSE57249",
]
NORM_METHODS: tuple[NormMethod, NormMethod] = ("log_cpm", "pearson")
