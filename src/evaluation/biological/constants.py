# Random gene sets drawn per cluster for its null distribution.
DEFAULT_STRATIFIED_SAMPLES = 100
DEFAULT_SEED = 3

# A cluster with fewer marker genes is scored but not tested: a 2D KDE fitted to a handful of
# genes is unreliable, and uniform random sets would make the test too lenient.
MIN_MARKER_GENES = 10

# A cell type counts as recovered when its best-matching cluster's Jaccard p-value is below this.
SIGNIFICANCE_LEVEL = 0.05

# Reference sets: DE markers of the ground-truth cell types (always built), and the marker databases
# a dataset can add in its config.yaml (biological_evaluation.marker_databases).
DE_MARKERS = "de_markers"
