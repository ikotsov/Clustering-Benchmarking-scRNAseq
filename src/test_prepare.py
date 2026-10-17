import pandas as pd
import pytest

from src.data_loading import load_csv_data, load_ground_truth_labels
from src.prepare import save_prepared_dataset


@pytest.fixture
def counts():
    return pd.DataFrame(
        {"GeneA": [1, 0, 5], "GeneB": [2, 3, 0]},
        index=["cell1", "cell2", "cell3"],
    )


def test_saves_counts_and_labels_in_counts_order(tmp_path, counts):
    labels = pd.Series({"cell3": "B", "cell1": "A", "cell2": "A"})

    save_prepared_dataset(counts, labels, tmp_path / "counts.csv.gz", tmp_path / "labels.csv")

    saved_counts = load_csv_data(tmp_path / "counts.csv.gz")
    pd.testing.assert_frame_equal(saved_counts, counts.rename_axis("cell_id"))
    saved_labels = load_ground_truth_labels(tmp_path / "labels.csv")
    assert saved_labels.index.tolist() == ["cell1", "cell2", "cell3"]
    assert saved_labels.tolist() == ["A", "A", "B"]


def test_rejects_unlabeled_cells(tmp_path, counts):
    labels = pd.Series({"cell1": "A", "cell2": "A"})
    with pytest.raises(ValueError, match="1 cells have no label"):
        save_prepared_dataset(counts, labels, tmp_path / "c.csv.gz", tmp_path / "l.csv")


def test_rejects_labels_for_unknown_cells(tmp_path, counts):
    labels = pd.Series({"cell1": "A", "cell2": "A", "cell3": "B", "cell4": "B"})
    with pytest.raises(ValueError, match="1 labeled cells are not in the counts"):
        save_prepared_dataset(counts, labels, tmp_path / "c.csv.gz", tmp_path / "l.csv")


def test_rejects_duplicate_cells(tmp_path, counts):
    duplicated = pd.concat([counts, counts.iloc[[0]]])
    labels = pd.Series({"cell1": "A", "cell2": "A", "cell3": "B"})
    with pytest.raises(ValueError, match="Duplicate cell IDs in the counts matrix"):
        save_prepared_dataset(duplicated, labels, tmp_path / "c.csv.gz", tmp_path / "l.csv")


def test_rejects_duplicate_genes(tmp_path, counts):
    duplicated = counts.set_axis(["GeneA", "GeneA"], axis=1)
    labels = pd.Series({"cell1": "A", "cell2": "A", "cell3": "B"})
    with pytest.raises(ValueError, match="Duplicate genes"):
        save_prepared_dataset(duplicated, labels, tmp_path / "c.csv.gz", tmp_path / "l.csv")
