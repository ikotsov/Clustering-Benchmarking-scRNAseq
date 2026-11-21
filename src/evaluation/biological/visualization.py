import json
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import DATASET_DIR, resolve


def load_biological_records(accession: str) -> pd.DataFrame:
    """Load the per-cluster records of every biological evaluation of a dataset into one table.

    One row per (run, enrichment set, cluster, cell type), with the run columns of results.tsv
    (algorithm, normalization, pca, params) taken from each file's path
    (outputs/runs/{norm_method}/{pca}/{params}/{algorithm}_biological.json).
    """
    runs_dir = resolve(DATASET_DIR, accession=accession) / "outputs" / "runs"
    tables = []
    for path in sorted(runs_dir.glob("*/*/*/*_biological.json")):
        normalization, pca, params = path.parts[-4:-1]
        with path.open() as f:
            records = json.load(f)["records"]
        tables.append(pd.DataFrame(records).assign(
            algorithm=path.name.removesuffix("_biological.json"),
            normalization=normalization, pca=pca, params=params,
        ))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def build_algorithm_celltype_heatmap_table(
    records: pd.DataFrame,
    metric: str = "jaccard",
    normalization: str | None = None,
    pca: str | None = None,
    params: str | None = None,
    enrichment_set: str | None = None,
    aggregate: str = "max",
) -> pd.DataFrame:
    """Build an algorithm x cell_type table for marker-alignment metrics.

    This summarizes biological comparison rows (see load_biological_records) by
    algorithm and reference cell_type, then pivots to a heatmap-ready matrix.
    aggregate="max" gives each cell type's best-matching cluster, as in the run summary.
    """
    required_cols = {"algorithm", "cell_type", metric}
    missing = required_cols - set(records.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    data = records.copy()
    if normalization is not None:
        data = data.loc[data["normalization"] == normalization]
    if pca is not None:
        data = data.loc[data["pca"] == pca]
    if params is not None:
        data = data.loc[data["params"] == params]
    if enrichment_set is not None:
        data = data.loc[data["enrichment_set"] == enrichment_set]

    if data.empty:
        return pd.DataFrame()

    agg_functions: dict[str, Callable[[pd.Series], float]] = {
        "mean": lambda s: float(pd.to_numeric(s, errors="coerce").mean()),
        "median": lambda s: float(pd.to_numeric(s, errors="coerce").median()),
        "max": lambda s: float(pd.to_numeric(s, errors="coerce").max()),
    }
    if aggregate not in agg_functions:
        supported = ", ".join(sorted(agg_functions))
        raise ValueError(
            f"Unsupported aggregate '{aggregate}'. Use one of: {supported}")

    value_col = f"{metric}_{aggregate}"
    grouped = (
        data.groupby(["algorithm", "cell_type"])[metric]
        .apply(agg_functions[aggregate])
        .reset_index(name=value_col)
    )

    table = grouped.pivot(
        index="algorithm", columns="cell_type", values=value_col)
    return table.sort_index(axis=0).sort_index(axis=1)


def plot_algorithm_celltype_heatmap(
    table: pd.DataFrame,
    title: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = ".3f",
) -> None:
    """Plot an algorithm x cell_type metric table as a heatmap."""
    if table.empty:
        raise ValueError(
            "Heatmap table is empty. Check your filters and input data.")

    values = table.to_numpy(dtype=float)

    fig_width = max(6.5, 0.9 * table.shape[1] + 2.5)
    fig_height = max(4.5, 0.55 * table.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Reference Cell Type")
    ax.set_ylabel("Algorithm")
    ax.set_title(title or "Algorithm vs Cell-Type Biological Alignment")

    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                label = "nan" if pd.isna(value) else format(value, fmt)
                ax.text(col, row, label, ha="center",
                        va="center", color="black")

    plt.tight_layout()
    plt.show()
