import json
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

from .types import BiologicalEvaluation, EnrichmentSetName, ReferenceSets
from src.types import PathLike

logger = logging.getLogger(__name__)


def save_reference_sets(output_path: PathLike, reference_sets: ReferenceSets) -> None:
    """Save a dataset's reference sets as JSON: {enrichment set: {cell type: [genes, sorted]}}."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {name: {cell_type: sorted(genes) for cell_type, genes in sets.items()}
             for name, sets in reference_sets.items()},
            f, indent=1,
        )
    logger.info(
        "Saved reference sets to %s: %s", output_path,
        "; ".join(f"{name}: {len(sets)} cell types" for name, sets in reference_sets.items()),
    )


def load_reference_sets(path: PathLike) -> ReferenceSets:
    with open(path) as f:
        loaded = json.load(f)
    return {
        cast(EnrichmentSetName, name): {cell_type: set(genes) for cell_type, genes in sets.items()}
        for name, sets in loaded.items()
    }


def save_biological_evaluation(
    output_path: PathLike,
    evaluation: BiologicalEvaluation,
    n_samples: int,
    seed: int,
) -> None:
    """
    Save the biological evaluation of one run into its own JSON file, next to the run's result:
        {
          "n_samples": 100, "seed": 42,
          "summary": {"bio_de_markers_jaccard": ..., ...},   # one value per results-table column
          "records": [...],                                  # one per (enrichment set, cell type, cluster)
          "cluster_markers": {"0": [genes], ...},
          "timestamp": "..."
        }
    NaN (statistics of untested clusters) is saved as null.
    """
    records = evaluation.records.astype(object).where(evaluation.records.notna(), None)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "n_samples": n_samples,
            "seed": seed,
            "summary": evaluation.summary,
            "records": records.to_dict(orient="records"),
            "cluster_markers": evaluation.cluster_markers,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=1, allow_nan=False)
    logger.info("Biological evaluation saved: %s (%s records)", output_path, len(evaluation.records))


def load_biological_summary(path: PathLike) -> dict[str, float | int]:
    """The bio_* columns of a run (see save_biological_evaluation)."""
    with open(path) as f:
        return json.load(f)["summary"]
