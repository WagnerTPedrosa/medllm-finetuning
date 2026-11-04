"""Build the multitask fine-tuning data (MedText, PubMedQA, MedMCQA, MedQA) in three steps.

Each step is a stage of pipelines/data/dvc.yaml and writes `<name>.jsonl` files
and a `stats.json`:

    uv run python -m medllm.data.finetune_data preprocess       # data/raw -> preprocessed_dir
    uv run python -m medllm.data.finetune_data split            # preprocessed_dir -> splits_dir
    uv run python -m medllm.data.finetune_data filter_length model=qwen3-1.7b   # -> processed_dir

1. preprocess: every source in the shared example format (see
   `medllm.data.preprocessing`); questions that need their options dropped; the
   official PubMedQA test questions kept apart and their articles removed
   from PQA-A.
2. split: validation sets held out (and MedText's test set); model-independent.
3. filter_length: examples longer than the model's context dropped (never
   truncated), with that model's tokenizer.

Every example appears once; the training mixture (repeats and samples per
epoch) is built when fine-tuning starts (see `medllm.data.mixture`).

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
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.data.download import load_raw
from medllm.data.preprocessing import preprocess
from medllm.data.splits import filter_length, split_examples

STEPS = ("preprocess", "split", "filter_length")


def save_splits(splits: dict[str, Dataset], output_dir: Path) -> None:
    """Write each split to `<output_dir>/<split>.jsonl`, one example per line (UTF-8).

    Args:
        splits: Splits by name.
        output_dir: Directory for the files (created if missing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_json(output_dir / f"{name}.jsonl", lines=True, force_ascii=False)


def load_finetune_data(processed_dir: str | Path) -> DatasetDict:
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


def write_step(splits: dict[str, Dataset], stats: dict[str, Any], output_dir: Path) -> None:
    """Save the splits as JSONL and the stats as `stats.json`, and print the stats."""
    save_splits(splits, output_dir)
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Saved to {output_dir}")


def run_preprocess(config: MultitaskConfig) -> None:
    """Step 1: read the downloaded datasets and write the examples to `multitask.preprocessed_dir`."""
    data = config.multitask
    output_dir = Path(data.preprocessed_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return

    raw_dir = Path(config.datasets.raw_dir)
    raw: dict[str, DatasetDict] = {name: load_raw(raw_dir / name) for name in config.datasets.sources}
    test_pmids: set[str] = set(json.loads(Path(data.pubmedqa_test_ground_truth).read_text()))
    examples, stats = preprocess(raw, test_pmids, data.num_proc, data.system_prompt)
    write_step(examples, stats, output_dir)


def run_split(config: MultitaskConfig) -> None:
    """Step 2: split the examples into `multitask.splits_dir`."""
    data = config.multitask
    output_dir = Path(data.splits_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return

    splits: dict[str, Dataset] = split_examples(dict(load_finetune_data(data.preprocessed_dir)), data, config.seed)
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
        dict(load_finetune_data(data.splits_dir)), tokenizer, config.data_filter.max_seq_len, data.num_proc
    )
    write_step(filtered, {"tokenizer": config.data_filter.tokenizer, **stats}, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multitask fine-tuning data, one step at a time.")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    if args.step == "preprocess":
        run_preprocess(config)
    elif args.step == "split":
        run_split(config)
    else:
        run_filter_length(config)


if __name__ == "__main__":
    main()
