import scprep
import pandas as pd
from typing import cast


def normalize_by_library_size(data, rescale=1_000_000) -> pd.DataFrame:
    """
    Normalizes counts per cell to sum to `rescale` (default CPM).

    If data is sparse, pseudocount must be 1 such that log(0 + pseudocount) = 0

    Parameters
    ----------
    data : array-like
    rescale : int, default=1_000_000

    Returns
    -------
    data_norm : array-like

    Examples
    --------
    >>> data (Total counts: S1=1000, S2=2000)
              Gene_A  Gene_B
    Sample_1     500     500
    Sample_2    1000    1000

    >>> normalize_library_size(data, rescale=1_000_000)
    # Both samples normalized to same depth
                 Gene_A     Gene_B
    Sample_1   500000.0   500000.0
    Sample_2   500000.0   500000.0
    """
    print(
        f"[Normalize]   Normalizing library size (CPM) with rescale={rescale:.0e}...")
    return scprep.normalize.library_size_normalize(data, rescale=rescale)


def log_transform(data, pseudocount=1) -> pd.DataFrame:
    """
    Applies log transformation: log(x + pseudocount).

    Parameters
    ----------
    data : array-like
    pseudocount : int, default=1

    Returns
    -------
    data_log : array-like

    Examples
    --------
    >>> data (CPM)
              Gene_A     Gene_B
    Sample_1   100.0        0.0

    >>> log_transform(data, pseudocount=1)
    # log(100 + 1) ~= 4.61, log(0 + 1) = 0
                Gene_A     Gene_B
    Sample_1  4.615121        0.0
    """
    print(f"[Transform]   Applying log transform (log{pseudocount}+x)...")

    data_log = scprep.transform.log(data, pseudocount=pseudocount)

    # Cast to DataFrame since scprep returns a generic array-like
    return cast(pd.DataFrame, data_log)
