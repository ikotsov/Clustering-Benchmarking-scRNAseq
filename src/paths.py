"""
Path conventions for the files the workflow reads and writes.

Templates are relative to the project root and use `{wildcard}` placeholders, so the Snakefile
can use them directly as rule patterns, while scripts and notebooks fill them in with `resolve`.
"""
from pathlib import Path

from src.constants import PROJECT_ROOT
from src.utils import get_pca_label


DATASET_DIR = "data/datasets/{accession}"
DATASET_CONFIG = DATASET_DIR + "/config.yaml"
RAW_COUNTS = DATASET_DIR + "/raw/{accession}.csv.gz"
GROUND_TRUTH_LABELS = DATASET_DIR + "/raw/ground_truth_labels.csv"
# {pca} is "pca" or "no_pca" (see get_pca_label).
PROCESSED_DATA = DATASET_DIR + "/processed/{norm_method}_{pca}.csv.gz"
HVGS = DATASET_DIR + "/processed/hvgs_{norm_method}.tsv"
CLUSTERING_PARAMS = DATASET_DIR + "/outputs/clustering_params.json"
RESULTS = DATASET_DIR + "/outputs/results.json"

GENE_ANNOTATION = "data/annotations/gene_annotation_{species}.csv"


def resolve(template: str, **wildcards: object) -> Path:
    """
    Fills in a path template and anchors it at the project root, e.g.
    resolve(PROCESSED_DATA, accession="E-MTAB-3321", norm_method="log_cpm", pca="pca")
    -> <project root>/data/datasets/E-MTAB-3321/processed/log_cpm_pca.csv.gz
    """
    return PROJECT_ROOT / template.format(**wildcards)


def processed_data_path(accession: str, norm_method: str, with_pca: bool) -> Path:
    return resolve(PROCESSED_DATA, accession=accession, norm_method=norm_method, pca=get_pca_label(with_pca))
