import pandas as pd
import scanpy as sc
from sklearn import cluster as sklearn_cluster
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, OPTICS, SpectralClustering

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


def agglomerative_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Clustering strategy using Agglomerative Clustering algorithm.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    '''
    n_clusters = kwargs.get("n_clusters", 5)
    linkage = kwargs.get("linkage", "ward")
    metric = kwargs.get("metric", "euclidean")

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
    )
    return pd.Series(model.fit_predict(data), index=data.index)


def hdbscan_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Clustering strategy using HDBSCAN algorithm.
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    '''
    hdbscan_cls = getattr(sklearn_cluster, "HDBSCAN")
    min_cluster_size = kwargs.get("min_cluster_size", 5)
    min_samples = kwargs.get("min_samples", None)
    cluster_selection_epsilon = kwargs.get("cluster_selection_epsilon", 0.0)
    model = hdbscan_cls(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    return pd.Series(model.fit_predict(data), index=data.index)


def leiden_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Clustering strategy using Scanpy Leiden algorithm.
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html
    '''
    resolution = float(kwargs.get("resolution", 1.0))

    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=SEED,
        # Using the igraph implementation and a fixed number of iterations can be significantly faster,
        # especially for larger datasets.
        # `directed` must be `False` to work with igraph’s implementation
        n_iterations=2,
        flavor='igraph',
        directed=False
    )

    labels = pd.Categorical(adata.obs['leiden']).codes
    return pd.Series(labels, index=data.index)
