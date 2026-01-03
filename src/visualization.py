import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    # Annotate genes (Max <= 1)
    bad_genes_count = (max_counts_before < 2).sum()
    axes[0].text(0.5, 0.9, f"Genes with max < 2:\n{bad_genes_count}",
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
    axes[1].axvline(1.5, color='black', linestyle='--',
                    linewidth=2, label='Cutoff (2)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# For neutral, unprocessed or dirty data - Grey is used to represent the baseline of the background.
GREY = '#95a5a6'
# For clean, processed, or filtered data - Green is associated with "pass" or "good".
GREEN = '#2ecc71'


def plot_filtering_effect(data_before: pd.DataFrame, data_after: pd.DataFrame, gene_list: list, metric_name: str, bins=50):
    """
    Calculates a specific metric (fraction of counts) for a set of genes 
    and plots the distribution before and after filtering.

    Args:
        data_before (pd.DataFrame): Dataframe before filtering.
        data_after (pd.DataFrame): Dataframe after filtering.
        gene_list (list): List of gene names to calculate the fraction for.
        metric_name (str): Label for the plot (e.g., 'Mitochondrial Fraction').
    """
    # Identify valid genes present in the dataset
    valid_genes = [g for g in gene_list if g in data_before.columns]

    if not valid_genes:
        print(
            f"Warning: No valid genes found for {metric_name} in the dataset.")
        return

    # Prepare the genes: calculate the fractions
    # Formula: Sum of specific genes / Total sum of all genes per cell
    values_before = data_before[valid_genes].sum(
        axis=1) / data_before.sum(axis=1)
    values_after = data_after[valid_genes].sum(axis=1) / data_after.sum(axis=1)

    # Plotting Logic
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot "Before" (Left) ---
    axes[0].hist(values_before, bins=bins, color=GREY,
                 edgecolor='black', alpha=0.7)
    axes[0].set_title(
        f"Before Filtering\n(n={len(values_before)})", fontweight='bold')
    axes[0].set_ylabel("Number of Cells")
    axes[0].set_xlabel(metric_name)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    stats_before = f"Max={np.max(values_before):.4f}\nMean={np.mean(values_before):.4f}"
    axes[0].text(0.95, 0.95, stats_before, transform=axes[0].transAxes,
                 va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Plot "After" (Right) ---
    axes[1].hist(values_after, bins=bins, color=GREEN,
                 edgecolor='black', alpha=0.7)
    axes[1].set_title(
        f"After Filtering\n(n={len(values_after)})", fontweight='bold')
    axes[1].set_xlabel(metric_name)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    stats_after = f"Max={np.max(values_after):.4f}\nMean={np.mean(values_after):.4f}"
    axes[1].text(0.95, 0.95, stats_after, transform=axes[1].transAxes,
                 va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
