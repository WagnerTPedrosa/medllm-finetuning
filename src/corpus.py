"""Prepare the PubMed corpus for continued pretraining, in two steps.

    uv run python -m src.corpus clean                       # model-independent
    uv run python -m src.corpus tokenize model=qwen3-1.7b   # one base model

clean (the raw JSONL shards are never modified), into `corpus.clean_dir`:
1. Remove articles used by the evaluation benchmark (decontamination).
2. Clean the text: drop templated administrative sections and inline citations.
3. Hold out a validation split, sampled within each MeSH stratum.
The documents are saved as Parquet (`train.parquet`, `validation.parquet`):
compressed, which matters at hundreds of thousands of full papers.

tokenize, into `corpus.processed_dir`:
4. Tokenize, append EOS to every document and pack into fixed-size blocks.

Each step writes a `stats.json` (what it removed or produced).
"""

import argparse
import json
import random
import re
from glob import glob
from itertools import chain
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import CorpusConfig, PretrainConfig, load_pretrain_config

HEADING_PREFIX = "## "
# Leading section numbers such as "5.", "2.1" or "IV."
SECTION_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+")
BOILERPLATE_HEADING = re.compile(
    r"""(
      acknowledge?ments?(\s+and\s+funding)?
    | funding(\s+(information|sources?|statement)|\s*(/|and)\s*support)?
    | sources?\s+of\s+funding
    | financial\s+(support|disclosures?)
    | role\s+of\s+the\s+(funders?|funding\s+sources?)
    | conflicts?\s+of\s+interests?(\s+(statement|disclosures?))?
    | (declaration\s+of\s+)?competing\s+interests?(\s+statement)?
    | declarations?(\s+of\s+interests?)?
    | disclosures?(\s+(statement|of\s+interests?))?
    | transparency\s+declaration
    | (dissemination|animal\s+studies)\s+declaration
    | disclaimer
    | (the\s+)?authors?'?s?\s+contributions?(\s+statement)?
    | credit\s+authorship\s+contribution\s+statement
    | contributors(hip)?
    | data(\s+and\s+code)?\s+availability(\s+statement)?
    | availability\s+of\s+(supporting\s+)?data(\s+and\s+materials?)?
    | (materials?|resources?)\s+availability
    | code\s+availability
    | (research\s+)?ethics(\s+(statement|declarations?|approval.*|and\s+consent.*|and\s+dissemination))?
    | statement\s+of\s+ethics
    | ethical\s+(approval|considerations?|statement|standards|declarations?|review|aspects).*
    | (informed\s+)?consent(\s+(statement|for\s+publication|to\s+participate))?
    | patient\s+consent.*
    | institutional\s+review\s+board\s+statement
    | trial\s+registration
    | patient\s+and\s+public\s+involvement
    | supplement(ary|al)\s+(information|materials?|data|files?)
    | supporting\s+information
    | electronic\s+supplementary\s+material
    | additional\s+files?
    | abbreviations
    | publisher'?s\s+note
    | (declaration\s+of\s+)?generative\s+ai.*
    )""",
    re.VERBOSE,
)
# [1], [1, 2], [3-5]
NUMERIC_CITATION = re.compile(r"\s?\[\d+(?:\s*[,–—-]\s*\d+)*\]")
# (Smith, 2020), (Smith et al., 2020), (Smith and Jones, 2019; Lee, 2021)
AUTHOR_YEAR_CITATION = re.compile(
    r"\s?\((?:[A-Z][\w'’-]+(?:\s+(?:et\s+al\.?|and|&)\s*(?:[A-Z][\w'’-]+)?)?,?\s+\d{4}[a-z]?(?:;\s*)?)+\)"
)
SPLIT_NAMES = ("train", "validation")
STEPS = ("clean", "tokenize")


def is_boilerplate_heading(heading: str) -> bool:
    """Whether a section heading names an administrative section (funding, ethics, ...).

    Args:
        heading: Heading text without the `## ` prefix.

    Returns:
        True for headings such as "5. Funding" or "Authors' contributions".
    """
    normalized: str = heading.strip().replace("’", "'")
    normalized = SECTION_NUMBER.sub("", normalized).rstrip(":. ").lower()
    return BOILERPLATE_HEADING.fullmatch(normalized) is not None


def remove_boilerplate_sections(text: str) -> str:
    """Drop administrative sections, from their heading up to the next heading.

    Args:
        text: Document text with `## Heading` lines and blank-line separated blocks.

    Returns:
        The text without those sections.
    """
    kept_blocks: list[str] = []
    skipping = False

    for block in text.split("\n\n"):
        if block.startswith(HEADING_PREFIX):
            skipping = is_boilerplate_heading(block[len(HEADING_PREFIX) :])
        if not skipping:
            kept_blocks.append(block)

    return "\n\n".join(kept_blocks)


def remove_citations(text: str) -> str:
    """Remove numeric ([1-3]) and author-year ((Smith et al., 2020)) citation markers."""
    text = NUMERIC_CITATION.sub("", text)
    return AUTHOR_YEAR_CITATION.sub("", text)


def clean_documents(
    batch: dict[str, list[Any]], remove_sections: bool, remove_cites: bool
) -> dict[str, list[Any]]:
    """Clean a batch of documents and record the length after each step (for `Dataset.map`).

    Returns:
        The cleaned `text` and the character counts `chars_raw`,
        `chars_after_sections` and `chars_clean`.
    """
    texts: list[str] = []
    chars_raw: list[int] = []
    chars_after_sections: list[int] = []
    chars_clean: list[int] = []

    for text in batch["text"]:
        chars_raw.append(len(text))
        if remove_sections:
            text = remove_boilerplate_sections(text)
        chars_after_sections.append(len(text))
        if remove_cites:
            text = remove_citations(text)
        chars_clean.append(len(text))
        texts.append(text)

    return {
        "text": texts,
        "chars_raw": chars_raw,
        "chars_after_sections": chars_after_sections,
        "chars_clean": chars_clean,
    }


def split_by_stratum(
    dataset: Dataset, validation_fraction: float, seed: int
) -> DatasetDict:
    """Hold out the same fraction of documents from every stratum.

    Args:
        dataset: Documents with a `stratum` column.
        validation_fraction: Fraction of each stratum used for validation.
        seed: Random seed (the split is reproducible).

    Returns:
        A `DatasetDict` with `train` and `validation` splits.
    """
    indices_by_stratum: dict[str, list[int]] = {}
    for index, stratum in enumerate(dataset["stratum"]):
        indices_by_stratum.setdefault(stratum, []).append(index)

    rng = random.Random(seed)
    validation_indices: list[int] = []
    for stratum in sorted(indices_by_stratum):
        indices: list[int] = indices_by_stratum[stratum]
        rng.shuffle(indices)
        validation_indices.extend(indices[: round(len(indices) * validation_fraction)])

    validation_set: set[int] = set(validation_indices)
    train_indices: list[int] = [
        i for i in range(len(dataset)) if i not in validation_set
    ]
    return DatasetDict(
        {
            "train": dataset.select(train_indices),
            "validation": dataset.select(sorted(validation_indices)),
        }
    )


def tokenize_documents(
    batch: dict[str, list[str]], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list[list[int]]]:
    """Tokenize documents and end each one with EOS, so documents stay separated after packing."""
    token_ids: list[list[int]] = tokenizer(batch["text"], add_special_tokens=False)[
        "input_ids"
    ]
    return {"input_ids": [ids + [tokenizer.eos_token_id] for ids in token_ids]}


def pack_blocks(
    batch: dict[str, list[list[int]]], block_size: int
) -> dict[str, list[list[int]]]:
    """Concatenate token sequences and cut them into blocks of `block_size`.

    The tokens left over at the end of the batch (fewer than one block) are dropped.
    """
    tokens: list[int] = list(chain.from_iterable(batch["input_ids"]))
    usable: int = len(tokens) - len(tokens) % block_size
    return {
        "input_ids": [
            tokens[start : start + block_size] for start in range(0, usable, block_size)
        ]
    }


def load_benchmark_pmids(dataset_id: str, subset: str) -> set[str]:
    """PMIDs of every article in a benchmark (all of its splits)."""
    benchmark = load_dataset(dataset_id, subset)
    return {str(pmid) for split in benchmark.values() for pmid in split["pubid"]}


def outside_benchmark(pmids: list[str | None], benchmark_pmids: set[str]) -> list[bool]:
    """Keep mask for a batch of PMIDs: False for articles used by the benchmark."""
    return [pmid not in benchmark_pmids for pmid in pmids]


def clean_corpus_documents(
    documents: Dataset, benchmark_pmids: set[str], corpus: CorpusConfig, seed: int
) -> tuple[DatasetDict, dict[str, Any]]:
    """Decontaminate, clean and split the documents.

    Args:
        documents: Raw documents with `pmid`, `pmcid`, `stratum` and `text`.
        benchmark_pmids: PMIDs of the evaluation benchmark, removed first.
        corpus: Cleaning options and validation fraction.
        seed: Random seed of the split.

    Returns:
        The `train` and `validation` documents, and what each step removed.
    """
    raw_documents: int = len(documents)
    documents = documents.filter(
        outside_benchmark,
        input_columns="pmid",
        batched=True,
        num_proc=corpus.num_proc,
        fn_kwargs={"benchmark_pmids": benchmark_pmids},
    )
    removed_benchmark: int = raw_documents - len(documents)

    documents = documents.map(
        clean_documents,
        batched=True,
        num_proc=corpus.num_proc,
        fn_kwargs={
            "remove_sections": corpus.remove_boilerplate_sections,
            "remove_cites": corpus.remove_citations,
        },
    )
    char_counts: dict[str, int] = {
        column: sum(documents[column])
        for column in ("chars_raw", "chars_after_sections", "chars_clean")
    }
    splits: DatasetDict = split_by_stratum(documents, corpus.validation_fraction, seed)

    stats: dict[str, Any] = {
        "seed": seed,
        "documents": {
            "raw": raw_documents,
            "removed_benchmark": removed_benchmark,
            "benchmark": f"{corpus.benchmark_dataset_id}/{corpus.benchmark_subset}",
            **{name: len(split) for name, split in splits.items()},
        },
        "cleaning": {
            "remove_boilerplate_sections": corpus.remove_boilerplate_sections,
            "remove_citations": corpus.remove_citations,
            "chars": char_counts,
            "removed_sections_pct": round(
                100 * (1 - char_counts["chars_after_sections"] / max(1, char_counts["chars_raw"])), 2
            ),
            "removed_citations_pct": round(
                100 * (1 - char_counts["chars_clean"] / max(1, char_counts["chars_after_sections"])), 2
            ),
        },
    }
    return splits, stats


def pack_corpus(
    splits: DatasetDict, tokenizer: PreTrainedTokenizerBase, block_size: int, num_proc: int
) -> tuple[DatasetDict, dict[str, Any]]:
    """Tokenize the documents and pack them into blocks of `block_size` tokens.

    Args:
        splits: `train` and `validation` documents with a `text` column.
        tokenizer: Tokenizer of the base model.
        block_size: Tokens per block (the model's context window).
        num_proc: Processes for `map`.

    Returns:
        The packed blocks (`input_ids`) and the token and block counts per split.
    """
    tokenized: DatasetDict = splits.map(
        tokenize_documents,
        batched=True,
        num_proc=num_proc,
        remove_columns=splits["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    packed: DatasetDict = tokenized.map(
        pack_blocks,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"block_size": block_size},
    )
    stats: dict[str, Any] = {
        "tokens": {name: sum(len(ids) for ids in split["input_ids"]) for name, split in tokenized.items()},
        "blocks": {name: len(split) for name, split in packed.items()},
        "packed_tokens": {name: len(split) * block_size for name, split in packed.items()},
    }
    return packed, stats


def run_clean_corpus(config: PretrainConfig) -> None:
    """Step 1: clean the raw shards into `corpus.clean_dir` (Parquet).

    Raises:
        FileNotFoundError: If `config.corpus.raw_dir` has no corpus shards.
    """
    corpus = config.corpus
    output_dir = Path(corpus.clean_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return

    shards: list[str] = sorted(glob(str(Path(corpus.raw_dir) / "corpus-*.jsonl")))
    if not shards:
        raise FileNotFoundError(f"No corpus-*.jsonl shards in {corpus.raw_dir}")
    documents: Dataset = load_dataset("json", data_files=shards, split="train")
    benchmark_pmids: set[str] = load_benchmark_pmids(corpus.benchmark_dataset_id, corpus.benchmark_subset)
    splits, stats = clean_corpus_documents(documents, benchmark_pmids, corpus, config.seed)

    output_dir.mkdir(parents=True)
    for name, split in splits.items():
        split.to_parquet(output_dir / f"{name}.parquet")
    stats = {"raw_dir": corpus.raw_dir, "shards": len(shards), **stats}
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (output_dir / "validation_pmcids.json").write_text(json.dumps(sorted(splits["validation"]["pmcid"])))
    print(json.dumps(stats, indent=2))
    print(f"Clean corpus saved to {output_dir}")


def run_tokenize_corpus(config: PretrainConfig) -> None:
    """Step 2: tokenize and pack the clean corpus into `corpus.processed_dir`."""
    corpus = config.corpus
    output_dir = Path(corpus.processed_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists")
        return

    splits: DatasetDict = load_dataset(
        "parquet", data_files={name: str(Path(corpus.clean_dir) / f"{name}.parquet") for name in SPLIT_NAMES}
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
    # Documents are packed into blocks later; the per-document length warning does not apply
    tokenizer.model_max_length = 10**9
    packed, stats = pack_corpus(splits, tokenizer, config.model.max_seq_len, corpus.num_proc)

    stats = {"clean_dir": corpus.clean_dir, "tokenizer": config.model.model_id, "block_size": config.model.max_seq_len, **stats}
    packed.save_to_disk(str(output_dir))
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Packed corpus saved to {output_dir}")


def load_pretrain_corpus(processed_dir: str | Path) -> DatasetDict:
    """Load the packed `train` and `validation` blocks written by `run_tokenize_corpus`."""
    return load_from_disk(str(processed_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the PubMed corpus, one step at a time.")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    args = parser.parse_args()

    config: PretrainConfig = load_pretrain_config(args.overrides)
    if args.step == "clean":
        run_clean_corpus(config)
    else:
        run_tokenize_corpus(config)


if __name__ == "__main__":
    main()
