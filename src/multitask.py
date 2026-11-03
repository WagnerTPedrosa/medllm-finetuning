"""Multitask fine-tuning data: MedText, PubMedQA and MedMCQA as prompt-completion pairs.

Every example has `prompt`, `completion`, `task` and `label` (the gold decision
or answer letter for the classification tasks, empty for MedText).

Splits:
    train          mixture of `training.max_epochs` epochs (see `build_mixture`)
    dev_*          model selection: MedText validation, 50 PubMedQA questions,
                   a MedMCQA sample held out from its train split; `dev_combined`
                   joins them for early stopping
    test_*         final results: MedText test, the official PubMedQA test set
                   (500), and the official MedMCQA validation set (its test set
                   has no labels)
"""

import argparse
import json
import random
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import MultitaskConfig, MultitaskDataConfig, load_multitask_config
from src.dataset_preparation import load_medtext
from src.formatting import format_prompt

EXAMPLE_COLUMNS = ["prompt", "completion", "task", "label"]
# One schema for every source, so splits from different datasets can be concatenated
EXAMPLE_FEATURES = Features({column: Value("string") for column in EXAMPLE_COLUMNS})
MEDMCQA_LETTERS = "ABCD"
PUBMEDQA_DECISIONS = ("yes", "no", "maybe")
MISSING_EXPLANATIONS = {"", "none"}

Example = dict[str, str]


def format_pubmedqa_prompt(contexts: list[str], labels: list[str], question: str) -> str:
    """Prompt with the abstract sections (labelled, e.g. `METHODS: ...`) and the question."""
    lines: list[str] = [
        f"{label}: {text}" if label else text for label, text in zip(labels, contexts)
    ]
    context: str = "\n".join(lines)
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer (yes, no or maybe):\n"


def format_pubmedqa_completion(decision: str, long_answer: str) -> str:
    """Decision first (so it can be scored on its own), then the explanation."""
    return f"{decision}. {long_answer}"


def format_medmcqa_prompt(question: str, options: list[str]) -> str:
    """Prompt with the question and the options lettered A-D."""
    lettered: str = "\n".join(
        f"{letter}. {option}" for letter, option in zip(MEDMCQA_LETTERS, options)
    )
    return f"Question:\n{question}\n\nOptions:\n{lettered}\n\nAnswer:\n"


def format_medmcqa_completion(
    answer_index: int, options: list[str], explanation: str | None
) -> str:
    """`B. <option>`, followed by the explanation when the dataset has one."""
    answer: str = f"{MEDMCQA_LETTERS[answer_index]}. {options[answer_index]}"
    if explanation is None or explanation.strip().lower() in MISSING_EXPLANATIONS:
        return answer
    return f"{answer}\n\nExplanation: {explanation.strip()}"


def medtext_example(row: dict[str, Any]) -> Example:
    """MedText row (`prompt`, `completion`) in the shared format (for `Dataset.map`)."""
    return {
        "prompt": format_prompt(row["prompt"]),
        "completion": row["completion"],
        "task": "medtext",
        "label": "",
    }


def pubmedqa_example(row: dict[str, Any]) -> Example:
    """PubMedQA row in the shared format (for `Dataset.map`)."""
    context: dict[str, list[str]] = row["context"]
    return {
        "prompt": format_pubmedqa_prompt(context["contexts"], context["labels"], row["question"]),
        "completion": format_pubmedqa_completion(row["final_decision"], row["long_answer"]),
        "task": "pubmedqa",
        "label": row["final_decision"],
    }


def medmcqa_example(row: dict[str, Any]) -> Example:
    """MedMCQA row in the shared format (for `Dataset.map`)."""
    options: list[str] = [row["opa"], row["opb"], row["opc"], row["opd"]]
    answer_index: int = int(row["cop"])
    return {
        "prompt": format_medmcqa_prompt(row["question"], options),
        "completion": format_medmcqa_completion(answer_index, options, row["exp"]),
        "task": "medmcqa",
        "label": MEDMCQA_LETTERS[answer_index],
    }


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


def balanced_sample_indices(labels: list[str], per_class: int, seed: int) -> list[int]:
    """Indices of up to `per_class` random items of each label."""
    indices_by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(index)

    rng = random.Random(seed)
    sample: list[int] = []
    for label in sorted(indices_by_label):
        indices: list[int] = indices_by_label[label]
        sample.extend(rng.sample(indices, min(per_class, len(indices))))
    return sorted(sample)


def epoch_chunks(num_items: int, per_epoch: int, num_epochs: int, seed: int) -> list[list[int]]:
    """Split shuffled passes over the items into per-epoch chunks.

    Every item is used once before any is repeated, and no chunk contains the
    same item twice.

    Raises:
        ValueError: If per_epoch is larger than num_items.
    """
    if per_epoch > num_items:
        raise ValueError(f"per_epoch ({per_epoch}) exceeds the number of items ({num_items})")

    rng = random.Random(seed)
    queue: deque[int] = deque()
    chunks: list[list[int]] = []

    for _ in range(num_epochs):
        chunk: list[int] = []
        chosen: set[int] = set()
        postponed: list[int] = []
        while len(chunk) < per_epoch:
            if not queue:
                order: list[int] = list(range(num_items))
                rng.shuffle(order)
                queue.extend(order)
            item: int = queue.popleft()
            if item in chosen:
                postponed.append(item)
                continue
            chunk.append(item)
            chosen.add(item)
        # Items already in this chunk are used first in the next one
        queue.extendleft(reversed(postponed))
        chunks.append(chunk)

    return chunks


def build_mixture(
    medtext: Dataset,
    pubmedqa_labeled: Dataset,
    pubmedqa_artificial: Dataset,
    medmcqa: Dataset,
    data_config: MultitaskDataConfig,
    num_epochs: int,
    seed: int,
) -> Dataset:
    """Training mixture for `num_epochs` epochs (shuffled by the trainer).

    Per epoch: MedText and PubMedQA-labeled repeated, PubMedQA-artificial
    balanced by decision (a new sample every epoch), and a new slice of MedMCQA.

    Args:
        medtext: MedText training examples.
        pubmedqa_labeled: PubMedQA (PQA-L) training examples.
        pubmedqa_artificial: PubMedQA (PQA-A) examples.
        medmcqa: MedMCQA training examples.
        data_config: Repeats and sample sizes.
        num_epochs: Number of epochs in the mixture.
        seed: Random seed.

    Returns:
        All epochs concatenated.
    """
    artificial_labels: list[str] = pubmedqa_artificial["label"]
    medmcqa_chunks: list[list[int]] = epoch_chunks(
        len(medmcqa), data_config.medmcqa_per_epoch, num_epochs, seed
    )

    parts: list[Dataset] = []
    for epoch in range(num_epochs):
        parts.extend([medtext] * data_config.medtext_repeats)
        parts.extend([pubmedqa_labeled] * data_config.pubmedqa_labeled_repeats)
        artificial_indices: list[int] = balanced_sample_indices(
            artificial_labels, data_config.pubmedqa_artificial_per_class, seed + epoch
        )
        parts.append(pubmedqa_artificial.select(artificial_indices))
        parts.append(medmcqa.select(medmcqa_chunks[epoch]))

    return concatenate_datasets(parts)


def split_pubmedqa_labeled(
    labeled: Dataset, test_pmids: set[str], validation_size: int, seed: int
) -> DatasetDict:
    """Official test questions, plus a validation split stratified by decision.

    Args:
        labeled: PQA-L rows with `pubid` and `final_decision`.
        test_pmids: PMIDs of the official test set.
        validation_size: Questions held out for validation from the rest.
        seed: Random seed.

    Returns:
        `train`, `validation` and `test` splits.
    """
    pmids: list[str] = [str(pmid) for pmid in labeled["pubid"]]
    test_indices: list[int] = [i for i, pmid in enumerate(pmids) if pmid in test_pmids]
    rest_indices: list[int] = [i for i, pmid in enumerate(pmids) if pmid not in test_pmids]

    rest: Dataset = labeled.select(rest_indices)
    validation_positions: list[int] = stratified_sample_indices(
        rest["final_decision"], validation_size, seed
    )
    validation_set: set[int] = set(validation_positions)
    train_positions: list[int] = [i for i in range(len(rest)) if i not in validation_set]

    return DatasetDict(
        {
            "train": rest.select(train_positions),
            "validation": rest.select(validation_positions),
            "test": labeled.select(test_indices),
        }
    )


def token_lengths(batch: dict[str, list[str]], tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    """Tokens per example: prompt + completion + EOS (for `Dataset.map`)."""
    prompts = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
    completions = tokenizer(batch["completion"], add_special_tokens=False)["input_ids"]
    return {"n_tokens": [len(p) + len(c) + 1 for p, c in zip(prompts, completions)]}


def fits_context(n_tokens: list[int], max_seq_len: int) -> list[bool]:
    """Keep mask for examples that fit the context window (for `Dataset.filter`)."""
    return [count <= max_seq_len for count in n_tokens]


def to_examples(
    dataset: Dataset, example_fn: Callable[[dict[str, Any]], Example], num_proc: int
) -> Dataset:
    """Map rows to the shared example format, dropping every other column."""
    return dataset.map(
        example_fn,
        remove_columns=dataset.column_names,
        features=EXAMPLE_FEATURES,
        num_proc=num_proc,
    )


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


def outside_pmids(pmids: list[int], excluded: set[str]) -> list[bool]:
    """Keep mask for rows whose PMID is not in `excluded` (for `Dataset.filter`)."""
    return [str(pmid) not in excluded for pmid in pmids]


def prepare_multitask(config: MultitaskConfig) -> DatasetDict | None:
    """Build and save every multitask split, plus `stats.json`.

    Args:
        config: Multitask configuration.

    Returns:
        The saved splits, or None if `multitask.processed_dir` already exists.
    """
    data = config.multitask
    output_dir = Path(data.processed_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return None

    # The same tokenizer for every base model, so all of them see the same examples
    tokenizer = AutoTokenizer.from_pretrained(config.data_filter.tokenizer)
    tokenizer.model_max_length = 10**9

    # MedText
    medtext_train, medtext_validation, medtext_test = load_medtext(data.medtext_dir)
    medtext = {
        name: to_examples(split, medtext_example, data.num_proc)
        for name, split in (
            ("train", medtext_train),
            ("validation", medtext_validation),
            ("test", medtext_test),
        )
    }

    # PubMedQA: official test set, the rest split into train/validation
    test_truth: dict[str, str] = json.loads(Path(data.pubmedqa_test_ground_truth).read_text())
    labeled: Dataset = load_dataset(data.pubmedqa_dataset_id, "pqa_labeled", split="train")
    labeled_splits: DatasetDict = split_pubmedqa_labeled(
        labeled, set(test_truth), data.pubmedqa_validation_size, config.seed
    )
    pubmedqa = {
        name: to_examples(split, pubmedqa_example, data.num_proc)
        for name, split in labeled_splits.items()
    }
    artificial_raw: Dataset = load_dataset(
        data.pubmedqa_dataset_id, "pqa_artificial", split="train"
    )
    labeled_pmids: set[str] = {str(pmid) for pmid in labeled["pubid"]}
    artificial_raw = artificial_raw.filter(
        outside_pmids,
        input_columns="pubid",
        batched=True,
        num_proc=data.num_proc,
        fn_kwargs={"excluded": labeled_pmids},
    )
    pubmedqa_artificial: Dataset = to_examples(artificial_raw, pubmedqa_example, data.num_proc)

    # MedMCQA: dev held out from train; the official validation set is the test set
    medmcqa_raw: DatasetDict = load_dataset(data.medmcqa_dataset_id)
    train_dev: DatasetDict = medmcqa_raw["train"].train_test_split(
        test_size=data.medmcqa_dev_fraction, seed=config.seed
    )
    medmcqa = {
        "train": to_examples(train_dev["train"], medmcqa_example, data.num_proc),
        "dev": to_examples(train_dev["test"], medmcqa_example, data.num_proc),
        "test": to_examples(medmcqa_raw["validation"], medmcqa_example, data.num_proc),
    }

    max_len: int = config.data_filter.max_seq_len
    dropped: dict[str, int] = {}
    named_sets: dict[str, Dataset] = {
        "medtext_train": medtext["train"],
        "pubmedqa_train": pubmedqa["train"],
        "pubmedqa_artificial": pubmedqa_artificial,
        "medmcqa_train": medmcqa["train"],
        "dev_medtext": medtext["validation"],
        "dev_pubmedqa": pubmedqa["validation"],
        "dev_medmcqa_full": medmcqa["dev"],
        "test_medtext": medtext["test"],
        "test_pubmedqa": pubmedqa["test"],
        "test_medmcqa": medmcqa["test"],
    }
    for name in named_sets:
        named_sets[name], dropped[name] = drop_too_long(
            named_sets[name], tokenizer, max_len, data.num_proc
        )

    mixture: Dataset = build_mixture(
        medtext=named_sets["medtext_train"],
        pubmedqa_labeled=named_sets["pubmedqa_train"],
        pubmedqa_artificial=named_sets["pubmedqa_artificial"],
        medmcqa=named_sets["medmcqa_train"],
        data_config=data,
        num_epochs=data.mixture_epochs,
        seed=config.seed,
    )
    dev_medmcqa: Dataset = named_sets["dev_medmcqa_full"].select(range(data.dev_medmcqa_size))

    splits = DatasetDict(
        {
            "train": mixture,
            "dev_combined": concatenate_datasets(
                [named_sets["dev_medtext"], named_sets["dev_pubmedqa"], dev_medmcqa]
            ),
            "dev_medtext": named_sets["dev_medtext"],
            "dev_pubmedqa": named_sets["dev_pubmedqa"],
            "dev_medmcqa": dev_medmcqa,
            "test_medtext": named_sets["test_medtext"],
            "test_pubmedqa": named_sets["test_pubmedqa"],
            "test_medmcqa": named_sets["test_medmcqa"],
        }
    )

    epoch_size: int = len(mixture) // data.mixture_epochs
    stats: dict[str, Any] = {
        "seed": config.seed,
        "epochs_in_mixture": data.mixture_epochs,
        "examples_per_epoch": epoch_size,
        "rows": {name: len(split) for name, split in splits.items()},
        "train_tasks": {
            task: mixture["task"].count(task) for task in ("medtext", "pubmedqa", "medmcqa")
        },
        "source_sizes": {name: len(dataset) for name, dataset in named_sets.items()},
        "dropped_too_long": dropped,
        "test_labels": {
            name: {label: splits[name]["label"].count(label) for label in sorted(set(splits[name]["label"]))}
            for name in ("test_pubmedqa", "test_medmcqa")
        },
    }

    splits.save_to_disk(str(output_dir))
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Multitask data saved to {output_dir}")
    return splits


def load_multitask(processed_dir: str | Path) -> DatasetDict:
    """Load the splits written by `prepare_multitask`."""
    return load_from_disk(str(processed_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multitask fine-tuning data.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args()

    prepare_multitask(load_multitask_config(args.overrides))


if __name__ == "__main__":
    main()
