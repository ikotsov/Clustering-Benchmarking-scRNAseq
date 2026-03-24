import pandas as pd
from typing import Callable, Dict, Literal
from .strategies import (
    agglomerative_strategy,
    birch_strategy,
    kmeans_strategy,
    leiden_strategy,
    optics_strategy,
    spectral_strategy,
    hdbscan_strategy
)

ClusteringFunc = Callable[..., pd.Series]
ClusteringAlgorithm = Literal[
    "agglomerative",
    "birch",
    "hdbscan",
    "kmeans",
    "leiden",
    "optics",
    "spectral",
]

STRATEGY_REGISTRY: Dict[ClusteringAlgorithm, ClusteringFunc] = {
    "agglomerative": agglomerative_strategy,
    "birch": birch_strategy,
    "hdbscan": hdbscan_strategy,
    "kmeans": kmeans_strategy,
    "leiden": leiden_strategy,
    "optics": optics_strategy,
    "spectral": spectral_strategy,
}

AVAILABLE_ALGORITHMS: tuple[ClusteringAlgorithm, ...] = tuple(
    STRATEGY_REGISTRY.keys())


def get_clustering_strategy(name: ClusteringAlgorithm) -> ClusteringFunc:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Algorithm '{name}' not found. Available: {list(STRATEGY_REGISTRY.keys())}")

    return STRATEGY_REGISTRY[name]
