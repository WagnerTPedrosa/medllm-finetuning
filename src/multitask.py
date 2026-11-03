"""Multitask fine-tuning data: MedText, PubMedQA, MedMCQA and MedQA as prompt-completion pairs.

MedMCQA and MedQA questions are asked open-ended, without their options: the
completion is the text of the correct option (with the cleaned explanation for
MedMCQA). Questions that only make sense with the options ("All of the
following except", "Which is true about") are left out.

Every example has `prompt`, `completion`, `task`, `label` (the gold decision for
PubMedQA, the letter of the correct option for MedMCQA and MedQA, empty for
MedText) and `options` (JSON list of the option texts, used to score the open
questions as multiple choice; empty for the other tasks).

The data is built in three steps (stages of pipelines/data/dvc.yaml), each one
writing `<name>.jsonl` files and a `stats.json`:

    uv run python -m src.multitask build_examples   # data/raw -> examples_dir
    uv run python -m src.multitask split            # examples_dir -> splits_dir
    uv run python -m src.multitask filter_length model=qwen3-1.7b   # -> processed_dir

1. build_examples: every source in the shared example format; questions that
   need their options dropped; the official PubMedQA test questions kept apart
   and their articles removed from PQA-A.
2. split: validation sets held out (and MedText's test set); model-independent.
3. filter_length: examples longer than the model's context dropped (never
   truncated), with that model's tokenizer.

Every example appears once; the training mixture (repeats and samples per epoch)
is built from the `train_*` splits when fine-tuning starts (see `build_mixture`).

Splits:
    train_*        unique training examples per source: medtext, pubmedqa
                   (PQA-L), pubmedqa_artificial (PQA-A), medmcqa, medqa
    validation_*   model selection: MedText validation, 50 PubMedQA questions,
                   MedMCQA and MedQA samples held out from their train splits;
                   `validation_combined` joins them for early stopping. (Our
                   `validation_medmcqa` comes from MedMCQA train; MedMCQA's
                   official `validation` split is `test_medmcqa`.)
    test_*         final results: MedText test, the official PubMedQA test set
                   (500), the official MedMCQA validation set (its test set has
                   no labels) and the official MedQA (USMLE, 4 options) test set
"""

import argparse
import json
import random
import re
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
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import MixtureConfig, MultitaskConfig, MultitaskDataConfig, load_multitask_config
from src.dataset_preparation import clean_examples, split_dataset
from src.download import load_raw
from src.formatting import format_prompt

EXAMPLE_COLUMNS = ["prompt", "completion", "task", "label", "options"]
# One schema for every source, so splits from different datasets can be concatenated
EXAMPLE_FEATURES = Features({column: Value("string") for column in EXAMPLE_COLUMNS})
OPTION_LETTERS = "ABCD"
PUBMEDQA_DECISIONS = ("yes", "no", "maybe")
TASKS = ("medtext", "pubmedqa", "medmcqa", "medqa")
STEPS = ("build_examples", "split", "filter_length")
MISSING_EXPLANATIONS = {"", "none"}

# "Ans. is 'c' i.e., ...", "Ans. D: ...", "Answer is A (...)", "a. ...": the answer letter, restated
ANSWER_PREFIX = re.compile(
    r"^\s*(?:ans(?:wer)?\b\.?\s*(?:is\b)?\s*[:.\-]?\s*(?:[('\"]\s*[a-d]\s*[)'\"]|[a-d](?![a-z0-9+]))?"
    r"|\(?[a-d][.)]\s)",
    re.IGNORECASE,
)
# Punctuation, "i.e." and source bullets ("o ") between the restated answer and the explanation
LEADING_SEPARATORS = re.compile(r"^(?:\s+|[.,:;*\-]+|i\.?\s?e\b\.?|o(?=\s))+", re.IGNORECASE)
# A citation starts with "Ref:", "Ref-", "Ref.", "Reference:" (any case) or "Ref Harrison";
# "Normal reference value ..." is not one
CITATION = r"(?:(?<![A-Za-z])(?i:ref(?:erences?)?)\s*[:.\-]|\b(?:Ref|REF)\s+(?=[A-Z]))"
# "(Ref: Nelson's 20/e p 720)": removed where it is, the explanation often continues after it
PARENTHESIZED_REFERENCE = re.compile(r"\(\s*" + CITATION + r"[^)]*\)")
# "Ref: Pal GK ... 2011: 4: 32.Explanation: ...": the citation goes, the explanation stays
CITATION_BEFORE_EXPLANATION = re.compile(CITATION + r".*?(?i:explanation)\s*:\s*", re.DOTALL)
# Any other citation is cut to the end of the text
TEXTBOOK_REFERENCE = re.compile(r"\(?" + CITATION + r".*$", re.DOTALL)
# Open-ended questions
WHICH_OF_THE_FOLLOWING = re.compile(r"\b([Ww])hich of the following\b")
SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+|\n+")
NEGATIVE_STEM = re.compile(r"\b(?:except|not|false|incorrect|least|untrue|wrong)\b", re.IGNORECASE)
REFERS_TO_OPTIONS = re.compile(
    r"\b(?:all|none|both) of the (?:following|above)\b|\bthe following\b"
    r"|\btrue (?:about|regarding|of|for)\b|\bstatements?\b",
    re.IGNORECASE,
)
ANSWER_REFERS_TO_OPTIONS = re.compile(
    r"^\s*(?:all|none|both)\s+(?:of\s+)?(?:the\s+)?(?:above|these|them|options)\b"
    r"|^\s*both\s+[a-d1-4]\b|\b[a-d]\s*(?:&|and|,)\s*[a-d]\b|^\s*[a-d]\s*$",
    re.IGNORECASE,
)
# Explanations shorter than this after cleaning add nothing to the answer
MIN_EXPLANATION_WORDS = 3

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


def format_open_completion(answer: str, explanation: str | None) -> str:
    """The answer, followed by the explanation when there is one."""
    if explanation is None:
        return answer
    return f"{answer}\n\nExplanation: {explanation}"


def open_question(question: str) -> str:
    """Question without its reference to the options ("Which of the following is" -> "What is")."""
    return WHICH_OF_THE_FOLLOWING.sub(what_with_same_case, question)


def what_with_same_case(match: re.Match[str]) -> str:
    """Replacement for `WHICH_OF_THE_FOLLOWING` that keeps the capital letter."""
    return "What" if match.group(1) == "W" else "what"


def question_stem(question: str) -> str:
    """Last sentence of the question (the actual question after a clinical vignette)."""
    sentences: list[str] = [part for part in SENTENCE_BREAK.split(question.strip()) if part.strip()]
    return sentences[-1] if sentences else question


def needs_options(question: str, answer: str) -> bool:
    """Whether a multiple-choice question only makes sense together with its options.

    True for negative stems ("except", "not", "false", "least"), questions about
    the options themselves ("Which is true about", "The following are") and
    answers that point at other options ("All of the above", "Both a and b").
    Negations are only looked for in the last sentence, so a vignette that
    says "he does not smoke" is kept.
    """
    if NEGATIVE_STEM.search(question_stem(question)):
        return True
    if REFERS_TO_OPTIONS.search(question):
        return True
    return ANSWER_REFERS_TO_OPTIONS.search(answer) is not None


def clean_medmcqa_explanation(explanation: str | None, correct_option: str) -> str | None:
    """Explanation without the restated answer and the textbook references.

    About a third of the MedMCQA explanations start by restating the answer
    ("Ans. is 'c' i.e., Mite") and a third cite a textbook ("Ref: Harrison,
    18th edition, Page 878"); trained on them, a model learns to invent
    citations. References in parentheses are removed where they are; other
    references are cut from the marker to the end of the text, so the rare
    explanation that continues after one loses its tail.

    Args:
        explanation: Raw `exp` field (None or "None" when missing).
        correct_option: Text of the correct option, to remove where it is restated.

    Returns:
        The cleaned explanation, or None when fewer than `MIN_EXPLANATION_WORDS` words are left.
    """
    if explanation is None or explanation.strip().lower() in MISSING_EXPLANATIONS:
        return None

    text: str = explanation.strip()
    prefix = ANSWER_PREFIX.match(text)
    if prefix is not None:
        text = LEADING_SEPARATORS.sub("", text[prefix.end() :])
        option: str = correct_option.strip().lower()
        if text.lower().startswith(f"({option})"):
            text = text[len(option) + 2 :]
        elif option and text.lower().startswith(option):
            text = text[len(option) :]

    text = PARENTHESIZED_REFERENCE.sub(" ", text)
    text = CITATION_BEFORE_EXPLANATION.sub("", text)
    text = LEADING_SEPARATORS.sub("", text)
    text = TEXTBOOK_REFERENCE.sub("", text).strip()
    if len(text.split()) < MIN_EXPLANATION_WORDS:
        return None
    return text


def medtext_example(row: dict[str, Any]) -> Example:
    """MedText row (`prompt`, `completion`) in the shared format (for `Dataset.map`)."""
    return {
        "prompt": format_prompt(row["prompt"]),
        "completion": row["completion"],
        "task": "medtext",
        "label": "",
        "options": "",
    }


def pubmedqa_example(row: dict[str, Any]) -> Example:
    """PubMedQA row in the shared format (for `Dataset.map`)."""
    context: dict[str, list[str]] = row["context"]
    return {
        "prompt": format_pubmedqa_prompt(context["contexts"], context["labels"], row["question"]),
        "completion": format_pubmedqa_completion(row["final_decision"], row["long_answer"]),
        "task": "pubmedqa",
        "label": row["final_decision"],
        "options": "",
    }


def medmcqa_options(row: dict[str, Any]) -> list[str]:
    """The four option texts of a MedMCQA row, in A-D order."""
    return [row["opa"], row["opb"], row["opc"], row["opd"]]


def medmcqa_example(row: dict[str, Any]) -> Example:
    """MedMCQA row as an open question answered with the correct option (for `Dataset.map`)."""
    options: list[str] = medmcqa_options(row)
    answer: str = options[int(row["cop"])]
    return {
        "prompt": format_prompt(open_question(row["question"])),
        "completion": format_open_completion(answer, clean_medmcqa_explanation(row["exp"], answer)),
        "task": "medmcqa",
        "label": OPTION_LETTERS[int(row["cop"])],
        "options": json.dumps(options),
    }


def medqa_example(row: dict[str, Any]) -> Example:
    """MedQA (USMLE) row as an open question answered with the correct option (for `Dataset.map`)."""
    options: list[str] = [row["options"][letter] for letter in OPTION_LETTERS]
    letter: str = row["answer_idx"]
    return {
        "prompt": format_prompt(open_question(row["question"])),
        "completion": row["options"][letter],
        "task": "medqa",
        "label": letter,
        "options": json.dumps(options),
    }


def medmcqa_is_open(row: dict[str, Any]) -> bool:
    """Keep MedMCQA rows whose question stands without the options (for `Dataset.filter`)."""
    return not needs_options(open_question(row["question"]), medmcqa_options(row)[int(row["cop"])])


def medqa_is_open(row: dict[str, Any]) -> bool:
    """Keep MedQA rows whose question stands without the options (for `Dataset.filter`)."""
    return not needs_options(open_question(row["question"]), row["options"][row["answer_idx"]])


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


def build_mixture(splits: DatasetDict, mixture: MixtureConfig, seed: int) -> Dataset:
    """Training mixture for `mixture.epochs` epochs (shuffled by the trainer).

    Per epoch: MedText, PubMedQA-labeled and MedQA repeated, PubMedQA-artificial
    balanced by decision (a new sample every epoch), and a new slice of MedMCQA.

    Args:
        splits: The unique training examples (`train_medtext`, `train_pubmedqa`,
            `train_pubmedqa_artificial`, `train_medmcqa`, `train_medqa`).
        mixture: Epochs, repeats and sample sizes.
        seed: Random seed.

    Returns:
        All epochs concatenated.
    """
    artificial: Dataset = splits["train_pubmedqa_artificial"]
    medmcqa: Dataset = splits["train_medmcqa"]
    artificial_labels: list[str] = artificial["label"]
    medmcqa_chunks: list[list[int]] = epoch_chunks(
        len(medmcqa), mixture.medmcqa_per_epoch, mixture.epochs, seed
    )

    parts: list[Dataset] = []
    for epoch in range(mixture.epochs):
        parts.extend([splits["train_medtext"]] * mixture.medtext_repeats)
        parts.extend([splits["train_pubmedqa"]] * mixture.pubmedqa_labeled_repeats)
        artificial_indices: list[int] = balanced_sample_indices(
            artificial_labels, mixture.pubmedqa_artificial_per_class, seed + epoch
        )
        parts.append(artificial.select(artificial_indices))
        parts.append(medmcqa.select(medmcqa_chunks[epoch]))
        parts.extend([splits["train_medqa"]] * mixture.medqa_repeats)

    return concatenate_datasets(parts)


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


def build_examples(
    raw: dict[str, DatasetDict], test_pmids: set[str], num_proc: int
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    """Convert every raw source to the shared example format.

    MedText is stripped and deduplicated. The official PubMedQA test questions
    are kept apart, and every PQA-L article is removed from PQA-A. MedMCQA and
    MedQA questions that need their options are dropped; MedMCQA's official
    validation split is our test set (its test split has no answers).

    Args:
        raw: Downloaded splits by source (`medtext`, `pubmedqa_labeled`,
            `pubmedqa_artificial`, `medmcqa`, `medqa`).
        test_pmids: PMIDs of the official PubMedQA test set.
        num_proc: Processes for `map` and `filter`.

    Returns:
        Examples by name (`medtext`, `pubmedqa_train`, `pubmedqa_test`,
        `pubmedqa_artificial`, `medmcqa_train`, `medmcqa_test`, `medqa_train`,
        `medqa_test`) and what each step removed.
    """
    medtext_rows, medtext_counts = clean_examples(raw["medtext"]["train"].to_pandas())
    medtext: Dataset = Dataset.from_pandas(medtext_rows, preserve_index=False)

    labeled: Dataset = raw["pubmedqa_labeled"]["train"]
    labeled_pmids: list[str] = [str(pmid) for pmid in labeled["pubid"]]
    test_indices: list[int] = [i for i, pmid in enumerate(labeled_pmids) if pmid in test_pmids]
    train_indices: list[int] = [i for i, pmid in enumerate(labeled_pmids) if pmid not in test_pmids]
    artificial_raw: Dataset = raw["pubmedqa_artificial"]["train"]
    artificial: Dataset = artificial_raw.filter(
        outside_pmids,
        input_columns="pubid",
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"excluded": set(labeled_pmids)},
    )

    open_questions: dict[str, Dataset] = {}
    dropped_needs_options: dict[str, int] = {}
    for name, source, split, is_open in (
        ("medmcqa_train", "medmcqa", "train", medmcqa_is_open),
        ("medmcqa_test", "medmcqa", "validation", medmcqa_is_open),
        ("medqa_train", "medqa", "train", medqa_is_open),
        ("medqa_test", "medqa", "test", medqa_is_open),
    ):
        rows: Dataset = raw[source][split]
        open_questions[name] = rows.filter(is_open, num_proc=num_proc)
        dropped_needs_options[name] = len(rows) - len(open_questions[name])

    examples: dict[str, Dataset] = {
        "medtext": to_examples(medtext, medtext_example, num_proc),
        "pubmedqa_train": to_examples(labeled.select(train_indices), pubmedqa_example, num_proc),
        "pubmedqa_test": to_examples(labeled.select(test_indices), pubmedqa_example, num_proc),
        "pubmedqa_artificial": to_examples(artificial, pubmedqa_example, num_proc),
        "medmcqa_train": to_examples(open_questions["medmcqa_train"], medmcqa_example, num_proc),
        "medmcqa_test": to_examples(open_questions["medmcqa_test"], medmcqa_example, num_proc),
        "medqa_train": to_examples(open_questions["medqa_train"], medqa_example, num_proc),
        "medqa_test": to_examples(open_questions["medqa_test"], medqa_example, num_proc),
    }
    stats: dict[str, Any] = {
        "rows": {name: len(dataset) for name, dataset in examples.items()},
        "medtext_cleaning": medtext_counts,
        "pubmedqa_artificial_overlap_removed": len(artificial_raw) - len(artificial),
        "dropped_needs_options": dropped_needs_options,
    }
    return examples, stats


def split_examples(
    examples: dict[str, Dataset], data: MultitaskDataConfig, seed: int
) -> dict[str, Dataset]:
    """Hold out the validation sets (and MedText's test set) from the examples.

    Args:
        examples: Output of `build_examples`.
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


def write_step(splits: dict[str, Dataset], stats: dict[str, Any], output_dir: Path) -> None:
    """Save the splits as JSONL and the stats as `stats.json`, and print the stats."""
    save_splits(splits, output_dir)
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Saved to {output_dir}")


def run_build_examples(config: MultitaskConfig) -> None:
    """Step 1: read the downloaded datasets and write the examples to `multitask.examples_dir`."""
    data = config.multitask
    output_dir = Path(data.examples_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return

    raw_dir = Path(config.datasets.raw_dir)
    raw: dict[str, DatasetDict] = {name: load_raw(raw_dir / name) for name in config.datasets.sources}
    test_pmids: set[str] = set(json.loads(Path(data.pubmedqa_test_ground_truth).read_text()))
    examples, stats = build_examples(raw, test_pmids, data.num_proc)
    write_step(examples, stats, output_dir)


def run_split(config: MultitaskConfig) -> None:
    """Step 2: split the examples into `multitask.splits_dir`."""
    data = config.multitask
    output_dir = Path(data.splits_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return

    splits: dict[str, Dataset] = split_examples(dict(load_multitask(data.examples_dir)), data, config.seed)
    stats: dict[str, Any] = {"seed": config.seed, "rows": {name: len(split) for name, split in splits.items()}}
    write_step(splits, stats, output_dir)


def run_filter_length(config: MultitaskConfig) -> None:
    """Step 3: filter the splits with the model's tokenizer into `multitask.processed_dir`."""
    data = config.multitask
    output_dir = Path(data.processed_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.data_filter.tokenizer)
    # Lengths are only measured here; this silences the warning for long examples
    tokenizer.model_max_length = 10**9
    filtered, stats = filter_length(
        dict(load_multitask(data.splits_dir)), tokenizer, config.data_filter.max_seq_len, data.num_proc
    )
    write_step(filtered, {"tokenizer": config.data_filter.tokenizer, **stats}, output_dir)


def save_splits(splits: dict[str, Dataset], output_dir: Path) -> None:
    """Write each split to `<output_dir>/<split>.jsonl`, one example per line (UTF-8).

    Args:
        splits: Splits by name.
        output_dir: Directory for the files (created if missing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_json(output_dir / f"{name}.jsonl", lines=True, force_ascii=False)


def load_multitask(processed_dir: str | Path) -> DatasetDict:
    """Load every `<split>.jsonl` file of a step's directory (examples, splits or processed).

    Args:
        processed_dir: Directory with the files.

    Returns:
        Every split, named after its file.

    Raises:
        FileNotFoundError: If the directory has no `.jsonl` files.
    """
    files: dict[str, str] = {path.stem: str(path) for path in sorted(Path(processed_dir).glob("*.jsonl"))}
    if not files:
        raise FileNotFoundError(f"No .jsonl splits in {processed_dir}")
    return load_dataset("json", data_files=files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multitask fine-tuning data, one step at a time.")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    if args.step == "build_examples":
        run_build_examples(config)
    elif args.step == "split":
        run_split(config)
    else:
        run_filter_length(config)


if __name__ == "__main__":
    main()
