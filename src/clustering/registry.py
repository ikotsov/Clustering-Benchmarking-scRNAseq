import pandas as pd
from typing import Callable, Dict
from .strategies import (
    agglomerative_strategy,
    birch_strategy,
    kmeans_strategy,
    leiden_strategy,
    optics_strategy,
    spectral_strategy,
)

ClusteringFunc = Callable[..., pd.Series]

STRATEGY_REGISTRY: Dict[str, ClusteringFunc] = {
    "agglomerative": agglomerative_strategy,
    "birch": birch_strategy,
    "kmeans": kmeans_strategy,
    "leiden": leiden_strategy,
    "optics": optics_strategy,
    "spectral": spectral_strategy,
    "spectral_clustering": spectral_strategy,
}


def get_clustering_strategy(name: str) -> ClusteringFunc:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Algorithm '{name}' not found. Available: {list(STRATEGY_REGISTRY.keys())}")

    return STRATEGY_REGISTRY[name]
