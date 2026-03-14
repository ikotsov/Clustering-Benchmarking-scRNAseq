import pandas as pd
from sklearn.cluster import Birch, KMeans, OPTICS, SpectralClustering

from src.constants import SEED


def kmeans_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Clustering strategy using K-Means algorithm.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 

    Selects initial cluster centroids using sampling based on an empirical probability distribution of the points’ contribution to the overall inertia.
    This technique speeds up convergence.

    '''
    n_clusters = kwargs.get("n_clusters", 5)
    model = KMeans(n_clusters=n_clusters, random_state=SEED)
    return pd.Series(model.fit_predict(data), index=data.index)


def spectral_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Clustering strategy using Spectral Clustering algorithm.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html 
    '''
    n_clusters = kwargs.get("n_clusters", 5)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        random_state=SEED,
    )
    return pd.Series(model.fit_predict(data), index=data.index)


def optics_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Clustering strategy using OPTICS algorithm.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html 
    '''
    min_samples = kwargs.get("min_samples", 5)
    model = OPTICS(min_samples=min_samples)
    return pd.Series(model.fit_predict(data), index=data.index)


def birch_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Clustering strategy using BIRCH algorithm.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
    '''
    n_clusters = kwargs.get("n_clusters", 5)
    threshold = kwargs.get("threshold", 0.5)
    branching_factor = kwargs.get("branching_factor", 50)
    model = Birch(
        n_clusters=n_clusters,
        threshold=threshold,
        branching_factor=branching_factor,
    )
    return pd.Series(model.fit_predict(data), index=data.index)
