"""Build a nested JSON index from the gene marker Excel files.

The output is grouped as:
    species -> db -> tissue -> cell type -> genes

Expected input files (located under `data/gene_markers/`):
- `Cell_marker_*.xlsx`
- `ScTypeDB_full.xlsx`

Output (default):
- `data/gene_markers/outputs/gene_marker_index.json`

For this repository, CellMarker files provide a real species value,
while ScType is species-agnostic; ScType entries are therefore stored
under the "all" species bucket.
"""

import json
import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd


logger = logging.getLogger(__name__)


GeneList = list[str]
CellTypeIndex = dict[str, GeneList]
TissueIndex = dict[str, CellTypeIndex]
DatabaseName = Literal["cell_marker", "sctype"]
SpeciesName = Literal["all", "human", "mouse"]
DatabaseIndex = dict[DatabaseName, TissueIndex]
GeneMarkerIndex = dict[SpeciesName, DatabaseIndex]

CELL_MARKER_DATABASE: DatabaseName = "cell_marker"
SCTYPE_DATABASE: DatabaseName = "sctype"
ALL_SPECIES: SpeciesName = "all"
HUMAN_SPECIES: SpeciesName = "human"
MOUSE_SPECIES: SpeciesName = "mouse"


CELL_MARKER_SCHEMA = {
    "species": "species",
    "db": CELL_MARKER_DATABASE,
    "tissue": "tissue_class",
    "cell_type": "cell_name",
    "gene": "marker",
}

SCTYPE_SCHEMA = {
    "species": ALL_SPECIES,
    "db": SCTYPE_DATABASE,
    "tissue": "tissueType",
    "cell_type": "cellName",
    # geneSymbolmore1 contains positive markers, while geneSymbolmore2 contains negative markers.
    "gene_columns": ("geneSymbolmore1", "geneSymbolmore2"),
}


def main() -> None:
    input_dir, output_path = get_marker_paths()
    write_gene_marker_index(input_dir, output_path)
    try:
        rel = output_path.relative_to(PROJECT_ROOT)
    except Exception:
        rel = output_path
    logger.info("Wrote gene marker index to %s", rel)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_NAME = "data"
GENE_MARKERS_DIR_NAME = "gene_markers"
GENE_MARKER_INDEX_FILENAME = "gene_marker_index.json"


def get_marker_paths() -> tuple[Path, Path]:
    data_dir = PROJECT_ROOT / DATA_DIR_NAME
    gene_markers_dir = data_dir / GENE_MARKERS_DIR_NAME
    outputs_dir = gene_markers_dir / "outputs"
    gene_marker_index_path = outputs_dir / GENE_MARKER_INDEX_FILENAME
    return gene_markers_dir, gene_marker_index_path


def write_gene_marker_index(input_dir: Path, output_path: Path) -> None:
    index = build_gene_marker_index(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def build_gene_marker_index(input_dir: Path) -> GeneMarkerIndex:
    index: GeneMarkerIndex = {}

    marker_paths = sorted(input_dir.rglob("*.xlsx"))
    if not marker_paths:
        raise FileNotFoundError(
            f"No .xlsx marker files found under {input_dir}")

    for marker_path in marker_paths:
        if marker_path.name.startswith("Cell_marker_"):
            _process_cell_marker(marker_path, index)
        elif marker_path.name == "ScTypeDB_full.xlsx":
            _process_sctype_marker(marker_path, index)

    return index


def _process_cell_marker(
    marker_path: Path,
    index: GeneMarkerIndex,
) -> None:
    data = _read_excel(marker_path)

    selected = data.loc[:, [
        CELL_MARKER_SCHEMA["species"],
        CELL_MARKER_SCHEMA["tissue"],
        CELL_MARKER_SCHEMA["cell_type"],
        CELL_MARKER_SCHEMA["gene"],
    ]].dropna(how="all")

    for species, tissue, cell_type, gene in selected.itertuples(index=False, name=None):
        species_key = _normalize_species(species)
        tissue_key = _normalize_label(tissue)
        cell_type_key = _normalize_label(cell_type)
        gene_key = _normalize_label(gene)
        if not tissue_key or not cell_type_key or not gene_key:
            continue

        _add_gene_entry(
            index,
            species_key,
            CELL_MARKER_DATABASE,
            tissue_key,
            cell_type_key,
            [gene_key],
        )


def _process_sctype_marker(
    marker_path: Path,
    index: GeneMarkerIndex,
) -> None:
    data = _read_excel(marker_path)

    selected = data.loc[:, [
        SCTYPE_SCHEMA["tissue"],
        SCTYPE_SCHEMA["cell_type"],
        # Use the primary gene column (geneSymbolmore1).
        SCTYPE_SCHEMA["gene_columns"][0],
    ]].dropna(how="all")

    for tissue, cell_type, gene_primary in selected.itertuples(index=False, name=None):
        tissue_key = _normalize_label(tissue)
        cell_type_key = _normalize_label(cell_type)
        if not tissue_key or not cell_type_key:
            continue

        genes = set()
        normalized = _normalize_label(gene_primary)
        if normalized:
            genes.update(
                token.strip()
                for token in normalized.split(",")
                if token.strip()
            )

        if not genes:
            continue

        _add_gene_entry(
            index,
            ALL_SPECIES,
            SCTYPE_DATABASE,
            tissue_key,
            cell_type_key,
            list(genes),
        )


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


EMPTY_STRING = ""


def _normalize_label(value: object) -> str:
    text = str(value).strip()
    if text == EMPTY_STRING or text.lower() == "nan":
        return EMPTY_STRING
    return text


def _normalize_species(value: object) -> SpeciesName:
    species = _normalize_label(value).lower()
    if species in {"human", "mouse"}:
        return cast(SpeciesName, species)
    return ALL_SPECIES


def _add_gene_entry(
    index: GeneMarkerIndex,
    species: SpeciesName,
    db: DatabaseName,
    tissue: str,
    cell_type: str,
    genes: GeneList,
) -> None:
    species_bucket = index.setdefault(species, {})
    db_bucket = species_bucket.setdefault(db, {})
    tissue_bucket = db_bucket.setdefault(tissue, {})

    existing = tissue_bucket.setdefault(cell_type, [])
    tissue_bucket[cell_type] = sorted({*existing, *genes})


if __name__ == "__main__":
    from src.logging_config import configure_logging

    configure_logging()
    main()
