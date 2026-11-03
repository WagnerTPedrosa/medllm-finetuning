from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

import src.download as download
from src.config import SourceDataset
from src.download import download_source, load_raw


class FakeHub:
    """Stands in for `load_dataset`, recording what was asked for."""

    def __init__(self, splits: DatasetDict) -> None:
        self.splits = splits
        self.calls: list[tuple[str, str | None, str]] = []

    def __call__(self, dataset_id: str, subset: str | None, revision: str) -> DatasetDict:
        self.calls.append((dataset_id, subset, revision))
        return self.splits


def test_download_source_saves_each_split_as_parquet_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = FakeHub(
        DatasetDict({"train": Dataset.from_dict({"q": ["a", "b"]}), "test": Dataset.from_dict({"q": ["c"]})})
    )
    monkeypatch.setattr(download, "load_dataset", hub)
    output_dir = tmp_path / "source"

    rows = download_source(SourceDataset("org/data", "sub", "abc123"), output_dir)

    assert hub.calls == [("org/data", "sub", "abc123")]
    assert rows == {"train": 2, "test": 1}
    assert sorted(path.name for path in output_dir.iterdir()) == ["test.parquet", "train.parquet"]
    monkeypatch.undo()
    assert load_raw(output_dir)["train"]["q"] == ["a", "b"]


def test_load_raw_names_the_directory_without_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_raw(tmp_path)
