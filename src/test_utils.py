import pandas as pd

from .utils import count_human_mt_genes, count_mouse_mt_genes


def test_count_mouse_mt_genes():
    df = pd.DataFrame(columns=["mt-Nd1", "mt-Co1", "MT-CO1", "Actb"])

    assert count_mouse_mt_genes(df) == 2


def test_count_human_mt_genes():
    df = pd.DataFrame(columns=["MT-CO1", "MT-ND2", "mt-Nd1", "ACTB"])

    assert count_human_mt_genes(df) == 2
