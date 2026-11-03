from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets

from src.config import MultitaskConfig, load_multitask_config
from src.multitask import (
    EXAMPLE_FEATURES,
    medtext_example,
    to_examples,
    balanced_sample_indices,
    build_mixture,
    epoch_chunks,
    format_medmcqa_completion,
    format_medmcqa_prompt,
    format_pubmedqa_completion,
    format_pubmedqa_prompt,
    medmcqa_example,
    pubmedqa_example,
    split_pubmedqa_labeled,
    stratified_sample_indices,
)



def load() -> MultitaskConfig:
    return load_multitask_config()


def pubmedqa_rows(count: int, decisions: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "pubid": list(range(count)),
            "question": [f"Question {i}?" for i in range(count)],
            "context": [{"contexts": ["Some text."], "labels": ["BACKGROUND"]}] * count,
            "long_answer": ["Because."] * count,
            "final_decision": [decisions[i % len(decisions)] for i in range(count)],
        }
    )


def test_format_pubmedqa_prompt_labels_context_sections() -> None:
    prompt = format_pubmedqa_prompt(
        ["Asthma is common.", "We studied 50 children."], ["BACKGROUND", "METHODS"], "Does X help?"
    )

    assert prompt == (
        "Context:\nBACKGROUND: Asthma is common.\nMETHODS: We studied 50 children.\n\n"
        "Question:\nDoes X help?\n\nAnswer (yes, no or maybe):\n"
    )


def test_format_pubmedqa_completion_starts_with_decision() -> None:
    assert format_pubmedqa_completion("no", "It did not help.") == "no. It did not help."


def test_format_medmcqa_prompt_lists_lettered_options() -> None:
    prompt = format_medmcqa_prompt("Which nerve?", ["Ulnar", "Radial", "Median", "Axillary"])

    assert prompt == (
        "Question:\nWhich nerve?\n\nOptions:\nA. Ulnar\nB. Radial\nC. Median\nD. Axillary"
        "\n\nAnswer:\n"
    )


def test_format_medmcqa_completion_adds_explanation_when_present() -> None:
    options = ["Ulnar", "Radial", "Median", "Axillary"]

    assert format_medmcqa_completion(1, options, "It supplies the extensors.") == (
        "B. Radial\n\nExplanation: It supplies the extensors."
    )


@pytest.mark.parametrize("explanation", [None, "", "None", "  "])
def test_format_medmcqa_completion_skips_missing_explanation(explanation: str | None) -> None:
    assert format_medmcqa_completion(2, ["a", "b", "c", "d"], explanation) == "C. c"


def test_pubmedqa_example_builds_prompt_completion_and_label() -> None:
    row = pubmedqa_rows(1, ["maybe"])[0]

    example = pubmedqa_example(row)

    assert example["task"] == "pubmedqa"
    assert example["label"] == "maybe"
    assert example["completion"] == "maybe. Because."
    assert example["prompt"].endswith("Answer (yes, no or maybe):\n")


def test_medmcqa_example_uses_answer_letter_as_label() -> None:
    row = {"question": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 3, "exp": None}

    example = medmcqa_example(row)

    assert example["label"] == "D"
    assert example["completion"] == "D. d"
    assert example["task"] == "medmcqa"


def test_stratified_sample_indices_keeps_label_proportions() -> None:
    labels = ["yes"] * 60 + ["no"] * 30 + ["maybe"] * 10

    indices = stratified_sample_indices(labels, size=10, seed=0)

    picked = [labels[i] for i in indices]
    assert len(indices) == 10
    assert (picked.count("yes"), picked.count("no"), picked.count("maybe")) == (6, 3, 1)


def test_split_pubmedqa_labeled_uses_official_test_and_stratified_validation() -> None:
    labeled = pubmedqa_rows(40, ["yes", "no"])
    test_pmids = {str(i) for i in range(20)}

    splits: DatasetDict = split_pubmedqa_labeled(labeled, test_pmids, validation_size=4, seed=0)

    assert sorted(splits["test"]["pubid"]) == list(range(20))
    assert len(splits["validation"]) == 4
    assert len(splits["train"]) == 16
    assert splits["validation"]["final_decision"].count("yes") == 2
    assert set(splits["train"]["pubid"]).isdisjoint(splits["validation"]["pubid"])


def test_balanced_sample_indices_caps_each_label() -> None:
    labels = ["yes"] * 90 + ["no"] * 10

    indices = balanced_sample_indices(labels, per_class=20, seed=0)

    picked = [labels[i] for i in indices]
    assert (picked.count("yes"), picked.count("no")) == (20, 10)


def test_epoch_chunks_uses_every_item_before_repeating() -> None:
    chunks = epoch_chunks(num_items=10, per_epoch=4, num_epochs=3, seed=0)

    assert [len(chunk) for chunk in chunks] == [4, 4, 4]
    first_pass = chunks[0] + chunks[1] + chunks[2][:2]
    assert sorted(first_pass) == list(range(10))
    assert all(len(set(chunk)) == len(chunk) for chunk in chunks)


def test_build_mixture_has_the_configured_examples_per_epoch() -> None:
    def task_rows(task: str, count: int, labels: list[str]) -> Dataset:
        return Dataset.from_dict(
            {
                "prompt": [f"{task} {i}" for i in range(count)],
                "completion": ["x"] * count,
                "task": [task] * count,
                "label": [labels[i % len(labels)] for i in range(count)],
            }
        )

    data_config = replace(
        load().multitask,
        medtext_repeats=2,
        pubmedqa_labeled_repeats=3,
        pubmedqa_artificial_per_class=5,
        medmcqa_per_epoch=7,
    )

    mixture = build_mixture(
        medtext=task_rows("medtext", 4, [""]),
        pubmedqa_labeled=task_rows("pubmedqa", 2, ["yes"]),
        pubmedqa_artificial=task_rows("pubmedqa_artificial", 30, ["yes", "yes", "no"]),
        medmcqa=task_rows("medmcqa", 10, ["A"]),
        data_config=data_config,
        num_epochs=2,
        seed=0,
    )

    tasks = mixture["task"]
    assert tasks.count("medtext") == 2 * 4 * 2
    assert tasks.count("pubmedqa") == 2 * 2 * 3
    assert tasks.count("pubmedqa_artificial") == 2 * (5 + 5)
    assert tasks.count("medmcqa") == 2 * 7
    assert len(mixture) == 16 + 12 + 20 + 14


def test_to_examples_uses_the_same_string_type_for_every_source() -> None:
    large = Features({"prompt": Value("large_string"), "completion": Value("large_string")})
    medtext = Dataset.from_dict({"prompt": ["Q?"], "completion": ["A."]}).cast(large)

    examples = to_examples(medtext, medtext_example, num_proc=1)

    assert examples.features == EXAMPLE_FEATURES
    assert len(concatenate_datasets([examples, examples])) == 2
