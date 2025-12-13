import scprep
import pandas as pd
from typing import cast
from src.constants import SEED

TARGET_CELLS = 5000


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
        sparse=True,
        # we know that gene symbols are unique in this dataset, so we can use them as labels
        gene_labels='symbol',
    )

    downsampled_data = scprep.select.subsample(
        raw_data, n=TARGET_CELLS, seed=SEED)

    # Silence the error because we know that scprep returns a pd.DataFrame in this case.
    return cast(pd.DataFrame, downsampled_data)
