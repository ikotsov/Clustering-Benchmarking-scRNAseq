"""
Loading and parsing of a dataset's config.yaml.

Kept free of heavy imports (pandas, scanpy), because the Snakefile imports it on every run.
"""
import os
from typing import cast, NotRequired, TypedDict

import yaml

from src.types import PathLike, Species


VALID_SPECIES: tuple[Species, Species] = ("human", "mouse")


class DatasetConfig(TypedDict):
    species: Species
    n_clusters: int
    preprocessing: NotRequired[dict[str, float | int]]
    clustering_params: NotRequired[dict[str, dict[str, object]]]


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
