import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import cast, List, Optional

from src.constants import SEED, GENE_MAGNITUDE_THRESHOLD

# To show cleanr and ready to use data - Blue is associated with stability.
BLUE = '#3498db'
# To warn about dirty or noisy data - Red is associated with attention.
RED = '#e74c3c'


def plot_gene_magnitude_distribution(data_before, data_after, x_limit=20):
    """
    Plots the distribution of MAXIMUM expression counts per gene to show
    the effect of filtering out low-magnitude genes.
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
    bad_genes_count = (max_counts_before < GENE_MAGNITUDE_THRESHOLD).sum()
    axes[0].text(0.5, 0.9, f"Genes with max < {GENE_MAGNITUDE_THRESHOLD}:\n{bad_genes_count}",
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
    axes[1].axvline(GENE_MAGNITUDE_THRESHOLD - 0.5, color='black', linestyle='--',
                    linewidth=2, label=f'Cutoff ({GENE_MAGNITUDE_THRESHOLD})')
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
    valid_genes = [g for g in gene_list if g in df.columns]
    if not valid_genes:
        return pd.Series(0, index=df.index)
    return df[valid_genes].sum(axis=1) / df.sum(axis=1).replace(0, 1)


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
        print(f"Warning: No valid genes found for {metric_name}")
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
    print(f"Mean variance: {gene_vars.mean():.2f}")
    print(f"Max variance:  {gene_vars.max():.2f}")
