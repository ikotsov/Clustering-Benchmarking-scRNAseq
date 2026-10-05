import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import cast, List, Optional

from src.constants import SEED, MIN_GENE_MAX_COUNT
from src.preprocessing.filters import apoptosis_genes, gene_set_fraction, matching_genes, mito_genes, rrna_genes
from src.preprocessing.types import PreprocessingConfig
from src.types import Species


logger = logging.getLogger(__name__)

# To show cleanr and ready to use data - Blue is associated with stability.
BLUE = '#3498db'
# To warn about dirty or noisy data - Red is associated with attention.
RED = '#e74c3c'


def plot_qc_overview(
    data: pd.DataFrame,
    count_threshold: Optional[float] = None,
    gene_threshold: Optional[float] = None,
    zoom_below: Optional[float] = None,
    color_by: Optional[pd.Series] = None,
    color_label: str = "Fraction of mitochondrial counts",
):
    """
    Plots the 2x2 cell QC overview from Luecken & Theis (2019), Fig. 2:
    (A) count depth histogram, (B) genes per cell histogram,
    (C) count depth ranked from high to low, (D) genes vs count depth coloured by color_by.

    Draw candidate thresholds on every panel to judge where to cut before
    setting them in the dataset config.

    Parameters
    ----------
    data : pd.DataFrame
        Raw counts (cells x genes), before any filtering.
    count_threshold, gene_threshold : float, optional
        Candidate minimum count depth / genes per cell, drawn as dashed lines.
    zoom_below : float, optional
        If given, panel A gets an inset histogram of the count depths below this value.
    color_by : pd.Series, optional
        Per-cell values for the colour of panel D. Defaults to the mitochondrial fraction;
        pass e.g. gene_set_fraction(data, rrna_genes(species)) to colour by rRNA instead.
    color_label : str
        Colour bar label for panel D.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_count_depth_histogram(
        data, threshold=count_threshold, zoom_below=zoom_below, ax=axes[0, 0])
    plot_genes_per_cell_histogram(
        data, threshold=gene_threshold, ax=axes[0, 1])
    plot_count_depth_rank(data, threshold=count_threshold, ax=axes[1, 0])
    plot_genes_vs_count_depth(
        data,
        color_by=color_by,
        color_label=color_label,
        count_threshold=count_threshold,
        gene_threshold=gene_threshold,
        ax=axes[1, 1],
    )

    for ax, letter in zip(axes.flat, "ABCD"):
        ax.text(-0.1, 1.05, letter, transform=ax.transAxes,
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_count_depth_histogram(data: pd.DataFrame, threshold: Optional[float] = None, zoom_below: Optional[float] = None, bins: int = 50, ax=None):
    """
    Histogram of count depth (total counts) per cell, optionally with a zoomed-in
    inset of the count depths below zoom_below.
    """
    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=(7, 5))

    count_depth = data.sum(axis=1)

    ax.hist(count_depth, bins=bins, color=BLUE, edgecolor='black', alpha=0.7)
    ax.set_title(f"Count depth per cell\n(n={len(count_depth)})", fontweight='bold')
    ax.set_xlabel("Count depth")
    ax.set_ylabel("Number of cells")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    draw_threshold(ax, threshold, vertical=True)

    if zoom_below is not None:
        inset = ax.inset_axes((0.45, 0.4, 0.5, 0.45))
        inset.hist(count_depth[count_depth < zoom_below], bins=bins,
                   color=BLUE, edgecolor='black', alpha=0.7)
        inset.set_title(f"Count depth < {zoom_below:g}", fontsize=9)
        inset.tick_params(labelsize=8)
        if threshold is not None and threshold < zoom_below:
            inset.axvline(threshold, color=RED, linestyle='--', linewidth=2)

    return finish_plot(ax, is_standalone)


def plot_genes_per_cell_histogram(data: pd.DataFrame, threshold: Optional[float] = None, bins: int = 50, ax=None):
    """
    Histogram of the number of genes detected (count > 0) per cell.
    """
    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=(7, 5))

    genes_per_cell = (data > 0).sum(axis=1)

    ax.hist(genes_per_cell, bins=bins, color=BLUE, edgecolor='black', alpha=0.7)
    ax.set_title(f"Genes detected per cell\n(n={len(genes_per_cell)})", fontweight='bold')
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Number of cells")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    draw_threshold(ax, threshold, vertical=True)

    return finish_plot(ax, is_standalone)


def plot_count_depth_rank(data: pd.DataFrame, threshold: Optional[float] = None, ax=None):
    """
    Count depth per cell sorted from high to low on log-log axes (the Cell Ranger
    "knee" plot). A sharp drop ("elbow") suggests where low-quality cells begin.
    """
    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=(7, 5))

    count_depth = np.sort(data.sum(axis=1).to_numpy())[::-1]
    cell_rank = np.arange(1, len(count_depth) + 1)

    ax.plot(cell_rank, count_depth, color=BLUE, linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Count depth by cell rank", fontweight='bold')
    ax.set_xlabel("Cell rank")
    ax.set_ylabel("Count depth")
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    draw_threshold(ax, threshold, vertical=False)

    return finish_plot(ax, is_standalone)


def plot_genes_vs_count_depth(
    data: pd.DataFrame,
    color_by: Optional[pd.Series] = None,
    color_label: str = "Fraction of mitochondrial counts",
    count_threshold: Optional[float] = None,
    gene_threshold: Optional[float] = None,
    ax=None,
):
    """
    Scatter of genes detected vs count depth per cell, coloured by a per-cell value
    (mitochondrial fraction by default), with both thresholds drawn to show their joint effect.
    """
    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=(8, 6))

    if color_by is None:
        color_by = gene_set_fraction(data, mito_genes(data))

    points = ax.scatter(
        data.sum(axis=1),
        (data > 0).sum(axis=1),
        c=color_by.reindex(data.index),
        cmap='viridis',
        s=12,
        alpha=0.8,
    )
    ax.figure.colorbar(points, ax=ax, label=color_label)
    ax.set_title("Genes detected vs count depth", fontweight='bold')
    ax.set_xlabel("Count depth")
    ax.set_ylabel("Number of genes")
    ax.grid(True, linestyle='--', alpha=0.3)
    draw_threshold(ax, count_threshold, vertical=True, name="Count threshold")
    draw_threshold(ax, gene_threshold, vertical=False, name="Gene threshold")

    return finish_plot(ax, is_standalone)


def plot_gene_set_qc(data: pd.DataFrame, species: Species = "human", config: PreprocessingConfig = PreprocessingConfig()):
    """
    Plots the mitochondrial, rRNA and apoptosis fractions against count depth in a row,
    each with its threshold from config. Cells above a threshold are the ones that filter removes.

    Parameters
    ----------
    data : pd.DataFrame
        Raw counts (cells x genes), before any filtering.
    species : Species
        Selects the rRNA and apoptosis gene sets.
    config : PreprocessingConfig
        Candidate thresholds, e.g. parse_preprocessing_config(load_dataset_config(dataset_dir)).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gene_sets = [
        ("Mitochondrial", mito_genes(data), config.mito_threshold),
        ("rRNA", rrna_genes(species), config.rrna_threshold),
        ("Apoptosis", apoptosis_genes(species), config.apoptosis_threshold),
    ]
    for ax, (name, gene_list, threshold) in zip(axes, gene_sets):
        plot_gene_set_fraction_vs_count_depth(
            data, gene_list, name, threshold=threshold, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_gene_set_fraction_vs_count_depth(data: pd.DataFrame, gene_list: List[str], name: str, threshold: Optional[float] = None, ax=None):
    """
    Scatter of the fraction of counts from gene_list against count depth per cell.
    Cells above the threshold are drawn in red. High fractions only in low-count cells
    point to damaged cells; high fractions across all count depths may be biology.
    """
    is_standalone = ax is None
    if is_standalone:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.set_xlabel("Count depth")
    ax.set_ylabel(f"Fraction of {name} counts")
    ax.grid(True, linestyle='--', alpha=0.3)

    n_genes = len(matching_genes(data, gene_list))
    if n_genes == 0:
        ax.set_title(f"{name} (0 genes found)", fontweight='bold')
        ax.text(0.5, 0.5, "No genes from this set found in the data",
                transform=ax.transAxes, ha='center', va='center', color=RED, fontweight='bold')
        return finish_plot(ax, is_standalone)

    count_depth = data.sum(axis=1)
    fraction = gene_set_fraction(data, gene_list)
    is_above = fraction > threshold if threshold is not None else pd.Series(False, index=data.index)

    ax.scatter(count_depth[~is_above], fraction[~is_above], color=BLUE, s=12, alpha=0.7)
    ax.scatter(count_depth[is_above], fraction[is_above], color=RED, s=12, alpha=0.9)

    title = f"{name} ({n_genes} genes found)"
    if threshold is not None:
        title += f"\n{int(is_above.sum())} of {len(fraction)} cells > {threshold:g}"
    ax.set_title(title, fontweight='bold')
    draw_threshold(ax, threshold, vertical=False)

    return finish_plot(ax, is_standalone)


def draw_threshold(ax, threshold: Optional[float], vertical: bool, name: str = "Threshold"):
    if threshold is None:
        return

    line = ax.axvline if vertical else ax.axhline
    line(threshold, color=RED, linestyle='--', linewidth=2,
         label=f"{name}: {threshold:g}")
    ax.legend()


def finish_plot(ax, is_standalone: bool):
    """Shows standalone figures; returns the axes when drawing into a caller's grid."""
    if is_standalone:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_gene_max_count_distribution(data_before, data_after, x_limit=20):
    """
    Plots the distribution of MAXIMUM expression counts per gene to show
    the effect of filtering out genes whose max count is below the cutoff.
    """
    # Calculate the MAX count for every gene (Column-wise max)
    max_counts_before = data_before.max(axis=0)
    max_counts_after = data_after.max(axis=0)

    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Set bins to align with integers (0, 1, 2...) for clarity
    bins = np.arange(0, x_limit + 1) - 0.5

    # --- Plot "Before" ---
    axes[0].hist(max_counts_before, bins=bins,
                 color=RED, edgecolor='black', alpha=0.7)
    axes[0].set_title(
        f"Before Filtering\n(Total Genes={len(data_before.columns)})", fontweight='bold')
    axes[0].set_xlabel("Max count per gene")
    axes[0].set_ylabel("Number of genes")
    axes[0].set_xticks(range(0, x_limit, 2))  # Ticks every 2 units
    axes[0].set_xlim(-0.5, x_limit)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Annotate genes (Max < threshold)
    bad_genes_count = (max_counts_before < MIN_GENE_MAX_COUNT).sum()
    axes[0].text(0.5, 0.9, f"Genes with max < {MIN_GENE_MAX_COUNT}:\n{bad_genes_count}",
                 transform=axes[0].transAxes, ha='center', color='red', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    # --- Plot "After" ---
    axes[1].hist(max_counts_after, bins=bins, color=BLUE,
                 edgecolor='black', alpha=0.7)
    axes[1].set_title(
        f"After Filtering\n(Total Genes={len(data_after.columns)})", fontweight='bold')
    axes[1].set_xlabel("Max count per gene")
    axes[1].set_xticks(range(0, x_limit, 2))
    axes[1].set_xlim(-0.5, x_limit)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    # Add a line to show the cutoff
    axes[1].axvline(MIN_GENE_MAX_COUNT - 0.5, color='black', linestyle='--',
                    linewidth=2, label=f'Cutoff ({MIN_GENE_MAX_COUNT})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# For neutral, unprocessed or dirty data - Grey is used to represent the baseline of the background.
GREY = '#95a5a6'
# For clean, processed, or filtered data - Green is associated with "pass" or "good".
GREEN = '#2ecc71'


def plot_filtering_effect(data_before, data_after, gene_list, metric_name):
    # Calculate values
    vals_before = calculate_gene_fraction(data_before, gene_list)
    vals_after = calculate_gene_fraction(data_after, gene_list)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reuse the generic plotter
    plot_metric_distribution(
        vals_before, f"Before: {metric_name}", color=GREY, ax=axes[0])
    plot_metric_distribution(
        vals_after, f"After: {metric_name}", color=GREEN, ax=axes[1])

    plt.tight_layout()
    plt.show()


def calculate_gene_fraction(df: pd.DataFrame, gene_list: list) -> pd.Series:
    """Helper to calculate the fraction of total counts for a gene set."""
    return gene_set_fraction(df, gene_list)


def plot_metric_distribution(values: pd.Series, title: str, cutoff: Optional[float] = None, color: str = BLUE, ax=None, label="Fraction of counts", bins=50):
    """
    Plots a single histogram for any numerical series.
    Does not require a gene list—just the final calculated values.
    """
    is_standalone = ax is None

    if is_standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(values, bins=bins, color=color, edgecolor='black', alpha=0.7)

    # Titles and Labels
    ax.set_title(f"{title}\n(n={len(values)})", fontweight='bold')
    ax.set_xlabel(label)
    ax.set_ylabel("Number of cells")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Optional Cutoff Line
    if cutoff is not None:
        ax.axvline(cutoff, color=RED, linestyle='--',
                   linewidth=2, label=f'Cutoff: {cutoff}')
        ax.legend()

    # Annotations
    stats_text = f"Max={values.max():.4f}\nMean={values.mean():.4f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if is_standalone:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_filtering_effect_violin(data_before: pd.DataFrame, data_after: pd.DataFrame, gene_list: list, metric_name: str):
    """
    Plots violin plots side-by-side to compare the distribution density before and after filtering.
    """
    # Identify valid genes
    valid_genes = [g for g in gene_list if g in data_before.columns]
    if not valid_genes:
        logger.warning("No valid genes found for %s", metric_name)
        return

    # Calculate metrics (Fractions)
    values_before = data_before[valid_genes].sum(
        axis=1) / data_before.sum(axis=1)
    values_after = data_after[valid_genes].sum(axis=1) / data_after.sum(axis=1)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the violin plot
    parts = ax.violinplot([values_before, values_after],
                          showmeans=True, showextrema=True)

    bodies = cast(List, parts['bodies'])

    # The 'bodies' key contains the colored area of the violin
    bodies[0].set_facecolor(GREY)
    bodies[0].set_edgecolor('black')
    bodies[0].set_alpha(0.7)

    bodies[1].set_facecolor(GREEN)
    bodies[1].set_edgecolor('black')
    bodies[1].set_alpha(0.7)

    # Style the lines (min/max/mean) to be standard black
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    # Labels and titles
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Before\n(n={len(values_before)})',
                       f'After\n(n={len(values_after)})'], fontweight='bold')
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribution shift: {metric_name}", fontweight='bold')

    # Add a horizontal grid for easier readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    # Add Stat Annotations (Optional but helpful)
    max_before = np.max(values_before)
    max_after = np.max(values_after)

    # Place text just above the max value of each violin
    ax.text(1, max_before, f"Max: {max_before:.4f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(2, max_after, f"Max: {max_after:.4f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_normalization_comparison(clean_data, normalized_data, n_cells=100):
    """
    Plots a side-by-side comparison of total transcripts per cell 
    before and after normalization.
    """
    # Take a subset (e.g., first 500 cells) to make the bars distinct and readable.
    indices = np.arange(n_cells)

    # Get values for the first N cells
    counts_before = clean_data.sum(axis=1).values[:n_cells]
    counts_after = normalized_data.sum(axis=1).values[:n_cells]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot "Before" ---
    axes[0].bar(indices, counts_before, width=1.0, alpha=0.9)
    axes[0].set_title(
        "Before normalization\n(variable sequencing depth)", fontweight='bold')
    axes[0].set_xlabel("Cell index")
    axes[0].set_ylabel("Total transcripts detected")
    axes[0].set_xlim(0, n_cells)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Add a text annotation to explain the jaggedness
    axes[0].text(0.5, 0.9, "Raw counts", transform=axes[0].transAxes,
                 ha='center', va='top', fontweight='bold', color='white',
                 bbox=dict(facecolor='black', alpha=0.3))

    # --- Plot "After" ---
    axes[1].bar(indices, counts_after, width=1.0, alpha=0.9)
    axes[1].set_title(
        f"After normalization\n(Scaled to CPM 1e6)", fontweight='bold')
    axes[1].set_xlabel("Cell index")
    axes[1].set_xlim(0, n_cells)
    # We match the Y-axis limit to show scale, or let it autoscale to show the flat line
    # (Autoscale is usually better here to see the line clearly)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    # Add annotation
    axes[1].text(0.5, 0.9, "Scaled counts", transform=axes[1].transAxes,
                 ha='center', va='top', fontweight='bold', color='white',
                 bbox=dict(facecolor='black', alpha=0.3))

    plt.tight_layout()
    plt.show()


def plot_log_transform_comparison(normalized_data, logged_data, sample_size=100000):
    """
    Plots a side-by-side histogram comparison of gene expression 
    before and after log transformation.
    """
    # --- 1. Data Preparation ---
    # We flatten the matrix to treat all gene counts as a single pool of numbers.
    # We sample 100,000 values to make plotting fast and avoid crashing.
    np.random.seed(SEED)  # For reproducibility
    # sample_size is passed as an argument

    # Flatten and sample "Before" (Normalized Data)
    flat_norm = normalized_data.values.flatten()
    if len(flat_norm) > sample_size:
        values_before = np.random.choice(flat_norm, sample_size, replace=False)
    else:
        values_before = flat_norm

    # Flatten and sample "After" (Logged Data)
    flat_log = logged_data.values.flatten()
    if len(flat_log) > sample_size:
        values_after = np.random.choice(flat_log, sample_size, replace=False)
    else:
        values_after = flat_log

    # Filter out pure zeros for a clearer view of the expression distribution
    # (Comment these out to see the zero-spike)
    values_before = values_before[values_before > 0]
    values_after = values_after[values_after > 0]

    # --- 2. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot "Before" (Linear Scale)
    axes[0].hist(values_before, bins=50, color=RED,
                 edgecolor='black', alpha=0.7)
    axes[0].set_title(
        f"Before Log Transform\n(Normalized CPM)", fontweight='bold')
    axes[0].set_xlabel("Expression Level (Counts)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Plot "After" (Log Scale)
    axes[1].hist(values_after, bins=50, color=BLUE,
                 edgecolor='black', alpha=0.7)
    axes[1].set_title(f"After Log Transform\n(Log1p CPM)", fontweight='bold')
    axes[1].set_xlabel("Log Expression Level")
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_pearson_diagnostic(pearsons_data):
    """
    Plots the mean-variance relationship of Pearson residuals to 
    verify variance stabilization.
    """
    # Calculate Mean and Variance
    gene_means = pearsons_data.mean(axis=0)
    gene_vars = pearsons_data.var(axis=0)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Scatter plot of genes
    plt.scatter(
        gene_means,
        gene_vars,
        alpha=0.4,    # Make points semi-transparent to see density
        s=15,         # Small marker size
        color=BLUE,
        label='HVGs (Top 3000)'
    )

    # Add a reference line at Variance - 1.0
    # This is the theoretical target for Pearson residuals
    plt.axhline(y=1.0, color='red', linestyle='--',
                linewidth=2, label="Target variance (1.0)")

    # 4. Formatting/Styling
    plt.title("Mean-Variance relationship (Pearson residuals)")
    plt.xlabel("Mean residual expression")
    plt.ylabel("Variance of residuals")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Ensure y-axis starts at 0 for clarity
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()

    # Health check
    logger.info("Mean variance: %.2f", gene_vars.mean())
    logger.info("Max variance: %.2f", gene_vars.max())
