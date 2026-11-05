"""Helpers of the `pubmedqa_test` harness task (see pubmedqa_test.yaml)."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

# PMIDs of the official PubMedQA test set
TEST_GROUND_TRUTH = Path(__file__).resolve().parents[4] / "data" / "benchmarks" / "pubmedqa_test_ground_truth.json"


def is_test_question(pubid: int, test_pmids: set[str]) -> bool:
    """Whether a PQA-L question belongs to the official test set (for `Dataset.filter`)."""
    return str(pubid) in test_pmids


def official_test(dataset: Dataset) -> Dataset:
    """Keep the 500 questions of the official test set."""
    test_pmids: set[str] = set(json.loads(TEST_GROUND_TRUTH.read_text()))
    return dataset.filter(is_test_question, input_columns="pubid", fn_kwargs={"test_pmids": test_pmids})


def doc_to_text(doc: dict[str, Any]) -> str:
    """The harness's PubMedQA prompt: the abstract, the question, then `Answer:`."""
    contexts: str = "\n".join(doc["context"]["contexts"])
    return f"Abstract: {contexts}\nQuestion: {doc['question']}\nAnswer:"
