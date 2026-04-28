import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_biological_comparison_records(results_json_path: str | Path) -> pd.DataFrame:
    """Load and flatten comparison records from biological_results.json.

    The returned dataframe contains run-level metadata columns (algorithm,
    normalization, with_pca, timestamp) merged with each row in
    comparison_records.
    """
    path = Path(results_json_path)
    with path.open("r") as f:
        payload = json.load(f)

    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(
            "Invalid biological results format: 'runs' must be a list.")

    flattened: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue

        run_metadata = {
            "dataset": payload.get("dataset"),
            "algorithm": run.get("algorithm"),
            "normalization": run.get("normalization"),
            "with_pca": run.get("with_pca"),
            "timestamp": run.get("timestamp"),
        }
        comparison_records = run.get("comparison_records", [])
        if not isinstance(comparison_records, list):
            continue

        for record in comparison_records:
            if not isinstance(record, dict):
                continue

            flattened.append({**run_metadata, **record})

    return pd.DataFrame(flattened)


def build_algorithm_celltype_heatmap_table(
    records: pd.DataFrame,
    metric: str = "jaccard",
    normalization: str | None = None,
    with_pca: bool | None = None,
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Build an algorithm x cell_type table for marker-alignment metrics.

    This summarizes biological comparison rows by algorithm and reference
    cell_type, then pivots to a heatmap-ready matrix.
    """
    required_cols = {"algorithm", "cell_type", metric}
    missing = required_cols - set(records.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    data = records.copy()
    if normalization is not None:
        data = data.loc[data["normalization"] == normalization]
    if with_pca is not None:
        data = data.loc[data["with_pca"] == with_pca]

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
