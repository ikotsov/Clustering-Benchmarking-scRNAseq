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
PROCESSED_DATA = DATASET_DIR + "/processed/{norm_method}_{pca}.csv.gz"
HVGS = DATASET_DIR + "/processed/hvgs_{norm_method}.tsv"
TUNING_RESULT = DATASET_DIR + "/outputs/tuning/{norm_method}/{algorithm}.json"
RUN_RESULT = DATASET_DIR + "/outputs/runs/{norm_method}/{pca}/{algorithm}.json"
CLUSTERING_PARAMS = DATASET_DIR + "/outputs/clustering_params.json"
RESULTS = DATASET_DIR + "/outputs/results.tsv"

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


def run_result_path(accession: str, norm_method: str, with_pca: bool, algorithm: str) -> Path:
    return resolve(RUN_RESULT, accession=accession, norm_method=norm_method, pca=get_pca_label(with_pca), algorithm=algorithm)
