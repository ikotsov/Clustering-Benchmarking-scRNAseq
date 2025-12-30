import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    axes[0].hist(values_before, bins=bins, color='#95a5a6',
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
    axes[1].hist(values_after, bins=bins, color='#2ecc71',
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
