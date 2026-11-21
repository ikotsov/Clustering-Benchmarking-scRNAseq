"""
Loading and parsing of a dataset's config.yaml.

Kept free of heavy imports (pandas, scanpy), because the Snakefile imports it on every run.
"""
import os
from typing import cast, NotRequired, TypedDict

import yaml

from src.types import PathLike, Species


VALID_SPECIES: tuple[Species, Species] = ("human", "mouse")
# Marker databases the biological evaluation can compare clusters with (see data/gene_markers/)
MARKER_DATABASES = ("cell_marker", "sctype")


class MarkerDatabaseConfig(TypedDict):
    # Tissues to take the cell types from, as named in data/gene_markers/outputs/gene_marker_index.json
    tissues: list[str]
    # Only these cell types of those tissues; all of them when left out
    cell_types: NotRequired[list[str]]


class BiologicalEvaluationConfig(TypedDict):
    marker_databases: NotRequired[dict[str, MarkerDatabaseConfig]]


class DatasetConfig(TypedDict):
    species: Species
    n_clusters: int
    preprocessing: NotRequired[dict[str, float | int]]
    clustering_params: NotRequired[dict[str, dict[str, object]]]
    biological_evaluation: NotRequired[BiologicalEvaluationConfig]


def load_dataset_config(config_path: PathLike) -> DatasetConfig:
    default_config: DatasetConfig = {
        "species": "human",
        "n_clusters": 0,
    }

    if not os.path.exists(config_path):
        return default_config

    with open(config_path, "r") as f:
        loaded = yaml.safe_load(f)

    if not isinstance(loaded, dict):
        return default_config

    return cast(DatasetConfig, loaded)


def parse_n_clusters(config: DatasetConfig) -> int | None:
    raw_n_clusters = config.get("n_clusters")
    if isinstance(raw_n_clusters, int) and raw_n_clusters > 0:
        return raw_n_clusters
    return None


def parse_species(config: DatasetConfig) -> Species:
    species = config.get("species")
    return species if species in VALID_SPECIES else "human"


def parse_marker_databases(config: DatasetConfig) -> dict[str, MarkerDatabaseConfig]:
    """
    The marker databases the biological evaluation compares the clusters with, besides the DE markers
    of the ground-truth cell types (always used). Set in the dataset's config.yaml, e.g.:

        biological_evaluation:
          marker_databases:
            cell_marker:
              tissues: [Blood]
              cell_types: [B cell, T cell, Monocyte]   # optional: all cell types of the tissues when left out
            sctype:
              tissues: [Immune system]

    The run's score averages over every cell type listed (see summarize_biological_evaluation), so list only
    cell types the dataset can contain. Empty when the dataset sets none.
    """
    section = config.get("biological_evaluation") or {}
    databases = section.get("marker_databases") or {}
    if not isinstance(databases, dict):
        raise ValueError("biological_evaluation.marker_databases must be a mapping of database name to settings")

    parsed: dict[str, MarkerDatabaseConfig] = {}
    for name, settings in databases.items():
        if name not in MARKER_DATABASES:
            raise ValueError(f"Unknown marker database '{name}'; expected one of {MARKER_DATABASES}")
        tissues = (settings or {}).get("tissues")
        if not isinstance(tissues, list) or not tissues or not all(isinstance(t, str) for t in tissues):
            raise ValueError(f"marker_databases.{name}.tissues must be a non-empty list of tissue names")
        entry: MarkerDatabaseConfig = {"tissues": tissues}
        cell_types = settings.get("cell_types")
        if cell_types is not None:
            if not isinstance(cell_types, list) or not all(isinstance(c, str) for c in cell_types):
                raise ValueError(f"marker_databases.{name}.cell_types must be a list of cell type names")
            entry["cell_types"] = cell_types
        parsed[name] = entry
    return parsed
