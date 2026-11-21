import numpy as np
import pandas as pd
import pytest

from .markers import build_database_marker_sets, build_de_marker_sets, compute_log_cpm


def _synthetic_counts(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """3 groups of 30 cells; group g expresses genes G{10g}..G{10g+9} 20x higher; 60 background genes."""
    rng = np.random.default_rng(seed)
    labels = pd.Series(np.repeat(["a", "b", "c"], 30), index=[f"cell{i}" for i in range(90)])
    rates = np.full((90, 90), 2.0)
    for g, group in enumerate("abc"):
        rates[(labels == group).to_numpy(), 10 * g:10 * g + 10] = 40.0
    counts = pd.DataFrame(rng.poisson(rates), index=labels.index, columns=[f"G{i}" for i in range(90)])
    return counts, labels


def test_build_de_marker_sets_finds_each_groups_genes():
    counts, labels = _synthetic_counts()

    markers = build_de_marker_sets(compute_log_cpm(counts), labels)

    for g, group in enumerate("abc"):
        assert {f"G{i}" for i in range(10 * g, 10 * g + 10)} <= markers[group]


def test_build_de_marker_sets_gives_a_one_cell_group_no_markers():
    counts, labels = _synthetic_counts()
    labels = labels.copy()
    labels.iloc[0] = "singleton"

    markers = build_de_marker_sets(compute_log_cpm(counts), labels)

    assert markers["singleton"] == set()
    assert len(markers["b"]) > 0


MARKER_INDEX = {
    "human": {"cell_marker": {
        "Blood": {"B cell": ["CD19", "MS4A1"], "T cell": ["CD3E"]},
        "Lung": {"B cell": ["CD79A"], "AT2 cell": ["SFTPC"]},
    }},
    "all": {"sctype": {"Immune system": {"B cells": ["CD19"]}}},
}


def test_build_database_marker_sets_maps_symbols_to_ensembl_ids_of_the_universe():
    annotation = pd.DataFrame({
        "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
        "gene_name": ["CD19", "Ms4a1", "CD3E", "SFTPC"],
    })
    universe = ["ENSG1", "ENSG2", "ENSG3"]  # SFTPC and CD79A are not measured

    sets = build_database_marker_sets(
        "cell_marker", "human", tissues=["Blood", "Lung"], cell_types=None,
        marker_index=MARKER_INDEX, universe=universe, annotation=annotation)

    # B cell: union over both tissues, matched case-insensitively; unmeasured genes dropped
    assert sets == {"AT2 cell": set(), "B cell": {"ENSG1", "ENSG2"}, "T cell": {"ENSG3"}}


def test_build_database_marker_sets_matches_symbol_universes_directly_and_filters_cell_types():
    sets = build_database_marker_sets(
        "cell_marker", "human", tissues=["Blood"], cell_types=["T cell"],
        marker_index=MARKER_INDEX, universe=["Cd3e", "Cd19"], annotation=None)

    assert sets == {"T cell": {"Cd3e"}}


def test_build_database_marker_sets_reads_sctype_from_the_all_species_bucket():
    sets = build_database_marker_sets(
        "sctype", "mouse", tissues=["Immune system"], cell_types=None,
        marker_index=MARKER_INDEX, universe=["Cd19"], annotation=None)

    assert sets == {"B cells": {"Cd19"}}


def test_build_database_marker_sets_rejects_unknown_tissues_and_cell_types():
    with pytest.raises(ValueError, match="unknown tissues"):
        build_database_marker_sets("cell_marker", "human", ["Brain"], None, MARKER_INDEX, ["CD19"])
    with pytest.raises(ValueError, match="not in tissues"):
        build_database_marker_sets("cell_marker", "human", ["Blood"], ["NK cell"], MARKER_INDEX, ["CD19"])
