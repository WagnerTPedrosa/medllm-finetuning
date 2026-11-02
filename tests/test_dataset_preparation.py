import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from src.dataset_preparation import (
    clean_examples,
    count_tokens,
    length_stats,
    load_medtext,
    split_dataset,
)


class CharacterTokenizer:
    """Test double: one token per character, so counts are easy to predict."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def test_clean_examples_strips_drops_empty_and_duplicates() -> None:
    raw = pd.DataFrame(
        {
            "Prompt": ["  a  ", "a", "", None, "b"],
            "Completion": ["x", "y", "z", "w", "  "],
        }
    )

    cleaned, counts = clean_examples(raw)

    assert cleaned.to_dict("records") == [{"prompt": "a", "completion": "x"}]
    assert counts == {"raw": 5, "after_empty_removal": 2, "after_dedup": 1}


def test_clean_examples_raises_for_missing_columns() -> None:
    with pytest.raises(ValueError, match="Completion"):
        clean_examples(pd.DataFrame({"Prompt": ["a"]}))


def test_count_tokens_includes_template_and_eos() -> None:
    template_length = len("Question:\n\n\nAnswer:\n")

    assert count_tokens("ab", "cde", CharacterTokenizer()) == template_length + 2 + 3 + 1


def test_split_dataset_creates_disjoint_80_10_10_splits() -> None:
    dataset = Dataset.from_dict({"prompt": [str(i) for i in range(100)]})

    splits: DatasetDict = split_dataset(dataset, holdout_fraction=0.2, seed=42)

    sizes = {name: len(split) for name, split in splits.items()}
    assert sizes == {"train": 80, "validation": 10, "test": 10}
    all_prompts = [prompt for split in splits.values() for prompt in split["prompt"]]
    assert len(set(all_prompts)) == 100


def test_split_dataset_is_reproducible() -> None:
    dataset = Dataset.from_dict({"prompt": [str(i) for i in range(50)]})

    first = split_dataset(dataset, holdout_fraction=0.2, seed=7)
    second = split_dataset(dataset, holdout_fraction=0.2, seed=7)

    assert first["test"]["prompt"] == second["test"]["prompt"]


def test_load_medtext_reads_saved_splits_with_extra_columns(tmp_path) -> None:
    splits = DatasetDict(
        {
            name: Dataset.from_dict(
                {"prompt": [name], "completion": ["answer"], "n_tokens": [3]}
            )
            for name in ("train", "validation", "test")
        }
    )
    splits.save_to_disk(str(tmp_path))

    train, validation, test = load_medtext(tmp_path)

    assert [split[0]["prompt"] for split in (train, validation, test)] == [
        "train",
        "validation",
        "test",
    ]
    assert all(
        split.column_names == ["prompt", "completion", "n_tokens"]
        for split in (train, validation, test)
    )


def test_length_stats_summarizes_lengths() -> None:
    stats = length_stats(list(range(1, 101)))

    assert stats == {"min": 1, "mean": 50.5, "p95": 95, "max": 100}


def test_length_stats_raises_for_empty_input() -> None:
    with pytest.raises(ValueError):
        length_stats([])
