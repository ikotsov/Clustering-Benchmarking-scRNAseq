import pytest

from src.dataset_config import parse_marker_databases


def test_parse_marker_databases_is_empty_without_the_section():
    assert parse_marker_databases({"species": "human", "n_clusters": 3}) == {}


def test_parse_marker_databases_reads_tissues_and_optional_cell_types():
    config = {"species": "human", "n_clusters": 3, "biological_evaluation": {"marker_databases": {
        "cell_marker": {"tissues": ["Blood"], "cell_types": ["B cell"]},
        "sctype": {"tissues": ["Immune system"]},
    }}}

    assert parse_marker_databases(config) == {
        "cell_marker": {"tissues": ["Blood"], "cell_types": ["B cell"]},
        "sctype": {"tissues": ["Immune system"]},
    }


@pytest.mark.parametrize("databases", [
    {"panglao": {"tissues": ["Blood"]}},
    {"cell_marker": {"tissues": []}},
    {"cell_marker": {"tissues": "Blood"}},
])
def test_parse_marker_databases_rejects_invalid_settings(databases):
    with pytest.raises(ValueError):
        parse_marker_databases({"species": "human", "n_clusters": 3, "biological_evaluation": {"marker_databases": databases}})
