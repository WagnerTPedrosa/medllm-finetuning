from dataclasses import replace

from datasets import Dataset, DatasetDict, concatenate_datasets

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.data.splits import (
    filter_length,
    split_dataset,
    split_examples,
    stratified_sample_indices,
)


def load() -> MultitaskConfig:
    return load_multitask_config()


def task_rows(task: str, count: int, labels: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [f"{task} {i}" for i in range(count)],
            "completion": ["x"] * count,
            "task": [task] * count,
            "label": [labels[i % len(labels)] for i in range(count)],
        }
    )


def test_stratified_sample_indices_keeps_label_proportions() -> None:
    labels = ["yes"] * 60 + ["no"] * 30 + ["maybe"] * 10

    indices = stratified_sample_indices(labels, size=10, seed=0)

    picked = [labels[i] for i in indices]
    assert len(indices) == 10
    assert (picked.count("yes"), picked.count("no"), picked.count("maybe")) == (6, 3, 1)


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


def test_split_examples_holds_out_validation_sets_and_keeps_the_tests() -> None:
    examples = {
        "medtext": task_rows("medtext", 20, [""]),
        "pubmedqa_train": task_rows("pubmedqa", 10, ["yes", "no"]),
        "pubmedqa_test": task_rows("pubmedqa_test", 3, ["yes"]),
        "pubmedqa_artificial": task_rows("pubmedqa_artificial", 5, ["yes"]),
        "medmcqa_train": task_rows("medmcqa", 12, ["A"]),
        "medmcqa_test": task_rows("medmcqa_test", 4, ["B"]),
        "medqa_train": task_rows("medqa", 9, ["C"]),
        "medqa_test": task_rows("medqa_test", 2, ["D"]),
    }
    data = replace(
        load().multitask,
        medtext_holdout_fraction=0.2,
        pubmedqa_validation_size=4,
        medmcqa_validation_size=3,
        medqa_validation_size=2,
    )

    splits = split_examples(examples, data, seed=0)

    assert {name: len(split) for name, split in splits.items()} == {
        "train_medtext": 16,
        "validation_medtext": 2,
        "test_medtext": 2,
        "train_pubmedqa": 6,
        "validation_pubmedqa": 4,
        "test_pubmedqa": 3,
        "train_pubmedqa_artificial": 5,
        "train_medmcqa": 9,
        "validation_medmcqa": 3,
        "test_medmcqa": 4,
        "train_medqa": 7,
        "validation_medqa": 2,
        "test_medqa": 2,
    }
    assert splits["validation_pubmedqa"]["label"].count("yes") == 2
    for task in ("medtext", "pubmedqa", "medmcqa", "medqa"):
        assert set(splits[f"train_{task}"]["prompt"]).isdisjoint(splits[f"validation_{task}"]["prompt"])
    assert split_examples(examples, data, seed=0)["validation_medqa"]["prompt"] == splits["validation_medqa"]["prompt"]


class WordTokenizer:
    """One token per word."""

    def __call__(self, texts: list[str], add_special_tokens: bool) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


def test_filter_length_drops_examples_longer_than_the_context_and_joins_the_validation_sets() -> None:
    splits = {
        f"{prefix}_{task}": task_rows(task, 2, ["A"])
        for prefix in ("train", "validation", "test")
        for task in ("medtext", "pubmedqa", "medmcqa", "medqa")
    }
    splits["train_medqa"] = concatenate_datasets(
        [splits["train_medqa"], task_rows("a very long medqa question", 1, ["A"])]
    )

    # "medqa 0" + "x" + EOS = 4 tokens; the long question has 7
    filtered, stats = filter_length(splits, WordTokenizer(), max_seq_len=5, num_proc=1)  # type: ignore[arg-type]

    assert len(filtered["train_medqa"]) == 2
    assert stats["dropped_too_long"]["train_medqa"] == 1
    assert len(filtered["validation_combined"]) == 8
    assert stats["rows"]["validation_combined"] == 8
    assert stats["test_labels"]["test_medqa"] == {"A": 2}
