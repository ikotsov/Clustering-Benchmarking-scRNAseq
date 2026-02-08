import scprep
import yaml
import os
import pandas as pd
from typing import cast, TypedDict, Union

from src.preprocessing.filters import Species


def load_csv_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, compression='gzip', index_col=0)


class DatasetConfig(TypedDict):
    species: Species
    n_clusters: int


def load_dataset_config(dataset_dir: str) -> Union[DatasetConfig, dict]:
    config_path = os.path.join(dataset_dir, "config.yaml")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_10x_data(data_path: str = "../data/") -> pd.DataFrame:
    """Load 10x data.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the 10x-formatted data.

    Returns
    -------
    raw_data : scprep.io.load_10X
        The loaded raw data as a sparse dataframe.
    """
    raw_data = scprep.io.load_10X(
        data_path,
        # sparse dataframes take up less memory than regular dataframes
        # if changed to False, downstream processing steps may need to be adjusted
        sparse=True,
        # we know that gene symbols are unique in this dataset, so we can use them as labels
        gene_labels='symbol',
    )

    # Silence the error because we know that scprep returns a pd.DataFrame in this case.
    return cast(pd.DataFrame, raw_data)
