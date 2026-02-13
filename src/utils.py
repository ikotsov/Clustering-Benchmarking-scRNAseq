import pandas as pd

DEFAULT_HVG_COUNT = 2000


def select_hvg_by_variance(data: pd.DataFrame, n_top_genes: int | None = None, percentile: float | None = None) -> pd.DataFrame:
    """
    Selects highly variable genes based on raw variance ranking.

    Args:
        data: Input DataFrame (Cells x Genes)
        n_top_genes: Specific number of genes to keep (e.g., 2000)
        percentile: Top percentage of genes to keep (e.g., 0.05 for 5%)

    Returns:
        DataFrame containing only the selected highly variable genes.
    """
    # Calculate variance for each gene (column-wise)
    gene_variances = data.var(axis=0).sort_values(ascending=False)

    # Determine how many genes to keep
    if percentile is not None:
        # Calculate number of genes based on percentage of total columns
        num_to_keep = int(len(gene_variances) * percentile)
        print(
            f"[HVG] Selecting top {percentile*100}% ({num_to_keep} genes) by variance.")
    elif n_top_genes is not None:
        num_to_keep = n_top_genes
        print(f"[HVG] Selecting top {num_to_keep} genes by variance.")
    else:
        # Default fallback if neither is provided
        num_to_keep = DEFAULT_HVG_COUNT
        print(
            f"[HVG] No criteria provided. Defaulting to top {DEFAULT_HVG_COUNT} genes.")

    # Get the names of the top genes
    top_genes = gene_variances.head(num_to_keep).index

    # Return the subsetted DataFrame
    return data[top_genes]


def extract_gene_subset(df, gene_list, subset_name="Subset"):
    """
    Intersects a dataframe's columns with a provided gene list 
    and returns the filtered dataframe with a summary.
    """
    # Convert the list of genes to a pandas Index for easy intersection
    gene_index = pd.Index(gene_list)
    # This creates a new Index containing ONLY the common gene names.
    overlapping = df.columns.intersection(gene_index)

    # Extract Data
    subset_df = df[overlapping]

    # Print Summary
    print(f"--- {subset_name} Summary ---")
    print(f"Total genes in dataset: {len(df.columns)}")
    print(f"Genes in target list:   {len(gene_list)}")
    print(f"Intersection found:     {len(overlapping)}")
    print("-" * (len(subset_name) + 12))

    return subset_df
