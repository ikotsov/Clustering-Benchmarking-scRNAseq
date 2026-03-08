import pandas as pd


def extract_gene_subset(df, gene_list, subset_name="Subset"):
    """
    Intersects a dataframe's columns with a provided gene list 
    and returns the filtered dataframe with a summary.
    """
    # Convert the list of genes to a pandas Index for easy intersection
    gene_index = pd.Index(gene_list)
    # This creates a new Index containing ONLY the common gene names.
    overlapping = df.columns.intersection(gene_index)

    count = len(overlapping)
    total_target = len(gene_list)

    # Print Summary
    print(f"{subset_name}: Matched {count} of {total_target} reference genes.")

    # Extract Data
    subset_df = df[overlapping]

    return subset_df


def count_mouse_mt_genes(data: pd.DataFrame) -> int:
    """
    Counts genes whose names start with 'mt-' (common mouse mitochondrial prefix).
    """
    return int(data.columns.str.startswith("mt-").sum())


def count_human_mt_genes(data: pd.DataFrame) -> int:
    """
    Counts genes whose names start with 'MT-' (common human mitochondrial prefix).
    """
    return int(data.columns.str.startswith("MT-").sum())
