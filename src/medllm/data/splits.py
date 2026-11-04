"""Train/validation/test splits and the context-length filter."""

import random
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from medllm.config import MultitaskDataConfig

TASKS = ("medtext", "pubmedqa", "medmcqa", "medqa")


def stratified_sample_indices(labels: list[str], size: int, seed: int) -> list[int]:
    """Indices of `size` items whose label proportions match the whole list.

    Quotas are rounded with the largest-remainder method, so they add up to `size`.

    Raises:
        ValueError: If size is larger than the number of items.
    """
    if size > len(labels):
        raise ValueError(f"Cannot sample {size} of {len(labels)} items")

    indices_by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(index)

    exact: dict[str, float] = {
        label: size * len(indices) / len(labels) for label, indices in indices_by_label.items()
    }
    quotas: dict[str, int] = {label: int(value) for label, value in exact.items()}
    remainders: dict[str, float] = {label: exact[label] - quotas[label] for label in exact}
    by_remainder: list[str] = sorted(remainders, key=remainders.__getitem__, reverse=True)
    for label in by_remainder[: size - sum(quotas.values())]:
        quotas[label] += 1

    rng = random.Random(seed)
    sample: list[int] = []
    for label in sorted(indices_by_label):
        sample.extend(rng.sample(indices_by_label[label], quotas[label]))
    return sorted(sample)


def split_dataset(dataset: Dataset, holdout_fraction: float, seed: int) -> DatasetDict:
    """Split into train, validation and test.

    The held-out fraction is divided equally between validation and test
    (for example 0.2 gives 80/10/10).

    Args:
        dataset: Examples to split.
        holdout_fraction: Fraction of examples kept out of training.
        seed: Random seed for reproducible splits.

    Returns:
        A `DatasetDict` with `train`, `validation` and `test` splits.
    """
    first_split: DatasetDict = dataset.train_test_split(
        test_size=holdout_fraction, seed=seed
    )
    holdout: DatasetDict = first_split["test"].train_test_split(test_size=0.5, seed=seed)

    return DatasetDict(
        {
            "train": first_split["train"],
            "validation": holdout["train"],
            "test": holdout["test"],
        }
    )


def split_examples(
    examples: dict[str, Dataset], data: MultitaskDataConfig, seed: int
) -> dict[str, Dataset]:
    """Hold out the validation sets (and MedText's test set) from the examples.

    Args:
        examples: Output of `preprocess`.
        data: Hold-out sizes.
        seed: Random seed.

    Returns:
        `train_*`, `validation_*` and `test_*` splits for every task
        (PQA-A is training data only: `train_pubmedqa_artificial`).
    """
    medtext: DatasetDict = split_dataset(examples["medtext"], data.medtext_holdout_fraction, seed)

    pubmedqa: Dataset = examples["pubmedqa_train"]
    validation_positions: list[int] = stratified_sample_indices(
        pubmedqa["label"], data.pubmedqa_validation_size, seed
    )
    validation_set: set[int] = set(validation_positions)
    train_positions: list[int] = [i for i in range(len(pubmedqa)) if i not in validation_set]

    medmcqa: DatasetDict = examples["medmcqa_train"].train_test_split(
        test_size=data.medmcqa_validation_size, seed=seed
    )
    medqa: DatasetDict = examples["medqa_train"].train_test_split(
        test_size=data.medqa_validation_size, seed=seed
    )

    return {
        "train_medtext": medtext["train"],
        "validation_medtext": medtext["validation"],
        "test_medtext": medtext["test"],
        "train_pubmedqa": pubmedqa.select(train_positions),
        "validation_pubmedqa": pubmedqa.select(validation_positions),
        "test_pubmedqa": examples["pubmedqa_test"],
        "train_pubmedqa_artificial": examples["pubmedqa_artificial"],
        "train_medmcqa": medmcqa["train"],
        "validation_medmcqa": medmcqa["test"],
        "test_medmcqa": examples["medmcqa_test"],
        "train_medqa": medqa["train"],
        "validation_medqa": medqa["test"],
        "test_medqa": examples["medqa_test"],
    }


def token_lengths(batch: dict[str, list[Any]], tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    """Tokens per conversation, chat template included, as the trainer tokenizes it (for `Dataset.map`)."""
    texts: list[str] = [
        tokenizer.apply_chat_template(prompt + completion, tokenize=False)
        for prompt, completion in zip(batch["prompt"], batch["completion"])
    ]
    return {"n_tokens": [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]}


def fits_context(n_tokens: list[int], max_seq_len: int) -> list[bool]:
    """Keep mask for examples that fit the context window (for `Dataset.filter`)."""
    return [count <= max_seq_len for count in n_tokens]


def drop_too_long(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_seq_len: int, num_proc: int
) -> tuple[Dataset, int]:
    """Remove examples longer than the context window.

    Returns:
        The kept examples (without the helper column) and how many were dropped.
    """
    with_lengths: Dataset = dataset.map(
        token_lengths, batched=True, num_proc=num_proc, fn_kwargs={"tokenizer": tokenizer}
    )
    kept: Dataset = with_lengths.filter(
        fits_context,
        input_columns="n_tokens",
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"max_seq_len": max_seq_len},
    )
    return kept.remove_columns("n_tokens"), len(dataset) - len(kept)


def filter_length(
    splits: dict[str, Dataset], tokenizer: PreTrainedTokenizerBase, max_seq_len: int, num_proc: int
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    """Drop examples longer than the context window and join the validation sets.

    Args:
        splits: Output of `split_examples`.
        tokenizer: Tokenizer of the model that will be fine-tuned.
        max_seq_len: Context window in tokens (prompt + completion + EOS).
        num_proc: Processes for `map` and `filter`.

    Returns:
        The kept examples (plus `validation_combined`, used for early
        stopping) and stats: rows, examples dropped per split, and the label
        counts of the test sets.
    """
    filtered: dict[str, Dataset] = {}
    dropped: dict[str, int] = {}
    for name, split in splits.items():
        filtered[name], dropped[name] = drop_too_long(split, tokenizer, max_seq_len, num_proc)
    filtered["validation_combined"] = concatenate_datasets(
        [filtered[f"validation_{task}"] for task in TASKS]
    )

    stats: dict[str, Any] = {
        "max_seq_len": max_seq_len,
        "rows": {name: len(split) for name, split in filtered.items()},
        "dropped_too_long": dropped,
        "test_labels": {
            name: {label: filtered[name]["label"].count(label) for label in sorted(set(filtered[name]["label"]))}
            for name in ("test_pubmedqa", "test_medmcqa", "test_medqa")
        },
    }
    return filtered, stats
