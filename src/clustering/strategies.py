import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering


def kmeans_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    n_clusters = kwargs.get("n_clusters", 5)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return pd.Series(model.fit_predict(data), index=data.index)


def spectral_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    n_clusters = kwargs.get("n_clusters", 5)
    model = SpectralClustering(
        n_clusters=n_clusters, affinity='nearest_neighbors')
    return pd.Series(model.fit_predict(data), index=data.index)
