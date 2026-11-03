import json
from pathlib import Path

import pytest
from datasets import Dataset

from medllm.data.multitask import load_multitask, save_splits


def task_rows(task: str, count: int, labels: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [f"{task} {i}" for i in range(count)],
            "completion": ["x"] * count,
            "task": [task] * count,
            "label": [labels[i % len(labels)] for i in range(count)],
        }
    )


def test_save_splits_writes_one_jsonl_per_split_that_load_multitask_reads_back(tmp_path: Path) -> None:
    splits = {
        "train_medqa": task_rows("medqa", 3, ["B"]),
        "test_medqa": Dataset.from_dict(
            {"prompt": ["Qual é a dose?"], "completion": ["5 µg"], "task": ["medqa"], "label": ["A"]}
        ),
    }

    save_splits(splits, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["test_medqa.jsonl", "train_medqa.jsonl"]
    first_line = (tmp_path / "test_medqa.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line)["completion"] == "5 µg"
    assert "µg" in first_line
    loaded = load_multitask(tmp_path)
    assert set(loaded) == {"train_medqa", "test_medqa"}
    assert loaded["train_medqa"].to_list() == splits["train_medqa"].to_list()


def test_load_multitask_names_the_directory_without_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_multitask(tmp_path)
