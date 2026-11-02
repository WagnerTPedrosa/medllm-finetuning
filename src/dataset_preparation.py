"""Download, clean, filter and split the MedText dataset."""

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from src.config import Config
from src.formatting import format_prompt

RAW_COLUMNS = ["Prompt", "Completion"]


class TextEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


def clean_examples(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strip whitespace, drop empty rows and duplicate prompts, rename columns.

    Args:
        raw: Raw MedText rows with `Prompt` and `Completion` columns.

    Returns:
        The cleaned rows with `prompt` and `completion` columns, and the number
        of rows left after each step.

    Raises:
        ValueError: If a required column is missing.
    """
    missing: set[str] = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    counts: dict[str, int] = {"raw": len(raw)}

    cleaned: pd.DataFrame = raw[RAW_COLUMNS].copy()
    for column in RAW_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned = cleaned.dropna(subset=RAW_COLUMNS)
    cleaned = cleaned[(cleaned["Prompt"] != "") & (cleaned["Completion"] != "")]
    counts["after_empty_removal"] = len(cleaned)

    cleaned = cleaned.drop_duplicates(subset=["Prompt"])
    counts["after_dedup"] = len(cleaned)

    cleaned = cleaned.rename(columns={"Prompt": "prompt", "Completion": "completion"})
    return cleaned.reset_index(drop=True), counts


def count_tokens(prompt: str, completion: str, tokenizer: TextEncoder) -> int:
    """Count the tokens of one training example: template + answer + EOS.

    Args:
        prompt: Raw question.
        completion: Reference answer.
        tokenizer: Tokenizer used for training.

    Returns:
        The number of tokens the model sees for this example.
    """
    prompt_tokens: int = len(tokenizer.encode(format_prompt(prompt)))
    completion_tokens: int = len(tokenizer.encode(completion))
    return prompt_tokens + completion_tokens + 1


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


def length_stats(lengths: list[int]) -> dict[str, float]:
    """Summarize sequence lengths.

    Args:
        lengths: Token counts.

    Returns:
        Minimum, mean (one decimal), 95th percentile and maximum.

    Raises:
        ValueError: If lengths is empty.
    """
    if not lengths:
        raise ValueError("lengths must not be empty")

    values = np.asarray(lengths)
    return {
        "min": int(values.min()),
        "mean": round(float(values.mean()), 1),
        "p95": int(np.percentile(values, 95)),
        "max": int(values.max()),
    }


def prepare_dataset(config: Config) -> DatasetDict | None:
    """Download MedText, clean it, drop over-long examples, split and save it.

    Writes the splits and a `stats.json` with row counts per step and token
    length statistics per split to `config.data.processed_dir`.

    Args:
        config: Experiment configuration.

    Returns:
        The saved splits, or None if the output directory already exists.
    """
    output_dir = Path(config.data.processed_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return None

    # Pin the revision so the processed data can be traced to an exact version
    revision: str = HfApi().dataset_info(config.data.dataset_id).sha
    raw: Dataset = load_dataset(config.data.dataset_id, split="train", revision=revision)

    examples, counts = clean_examples(raw.to_pandas())

    # Drop (rather than truncate) examples that do not fit the context window,
    # so the model never trains on cut-off answers
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
    examples["n_tokens"] = [
        count_tokens(prompt, completion, tokenizer)
        for prompt, completion in zip(examples["prompt"], examples["completion"])
    ]
    fits_context = examples["n_tokens"] <= config.model.max_seq_len
    examples = examples[fits_context]
    counts["after_length_filter"] = len(examples)

    splits: DatasetDict = split_dataset(
        Dataset.from_pandas(examples, preserve_index=False),
        config.data.holdout_fraction,
        config.seed,
    )

    stats: dict[str, Any] = {
        "dataset_id": config.data.dataset_id,
        "revision": revision,
        "seed": config.seed,
        "tokenizer": config.model.model_id,
        "max_seq_len": config.model.max_seq_len,
        "rows": counts,
        "removed_too_long": int((~fits_context).sum()),
        "splits": {
            name: {"rows": len(split), "n_tokens": length_stats(split["n_tokens"])}
            for name, split in splits.items()
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(output_dir))
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(splits)
    print(json.dumps(stats, indent=2))
    print(f"Dataset saved to {output_dir}")
    return splits


def load_medtext(processed_dir: str | Path) -> tuple[Dataset, Dataset, Dataset]:
    """Load the processed train, validation and test splits.

    Args:
        processed_dir: Directory written by `prepare_dataset`.

    Returns:
        The train, validation and test splits.
    """
    splits: DatasetDict = load_from_disk(str(processed_dir))
    return splits["train"], splits["validation"], splits["test"]
