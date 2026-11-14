import os

from typing_extensions import Literal

Species = Literal["human", "mouse"]
NormMethod = Literal["pearson", "log_cpm"]
PathLike = str | os.PathLike[str]

ExternalMetric = Literal["ari", "nmi", "jaccard"]
InternalMetric = Literal["silhouette", "calinski_harabasz", "davies_bouldin"]
