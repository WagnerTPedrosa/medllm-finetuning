import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets

from dataclasses import replace

from src.config import MixtureConfig, MultitaskConfig, load_multitask_config
from src.formatting import format_prompt
from src.multitask import (
    EXAMPLE_FEATURES,
    medtext_example,
    to_examples,
    balanced_sample_indices,
    build_mixture,
    epoch_chunks,
    format_open_completion,
    load_multitask,
    needs_options,
    open_question,
    format_pubmedqa_completion,
    format_pubmedqa_prompt,
    clean_medmcqa_explanation,
    medmcqa_example,
    medqa_example,
    pubmedqa_example,
    build_examples,
    filter_length,
    save_splits,
    split_examples,
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


def test_format_open_completion_adds_the_explanation_when_present() -> None:
    assert format_open_completion("Radial nerve", "It supplies the extensors.") == (
        "Radial nerve\n\nExplanation: It supplies the extensors."
    )
    assert format_open_completion("Radial nerve", None) == "Radial nerve"


def test_pubmedqa_example_builds_prompt_completion_and_label() -> None:
    row = pubmedqa_rows(1, ["maybe"])[0]

    example = pubmedqa_example(row)

    assert example["task"] == "pubmedqa"
    assert example["label"] == "maybe"
    assert example["completion"] == "maybe. Because."
    assert example["prompt"].endswith("Answer (yes, no or maybe):\n")


def test_medmcqa_example_asks_the_question_open_ended() -> None:
    row = {"question": "Thinning of cornea occurs in", "opa": "Glaucoma", "opb": "Keratoconus",
           "opc": "Uveitis", "opd": "Cataract", "cop": 1, "exp": None}

    example = medmcqa_example(row)

    assert example["prompt"] == format_prompt("Thinning of cornea occurs in")
    assert example["completion"] == "Keratoconus"
    assert example["label"] == "B"
    assert json.loads(example["options"]) == ["Glaucoma", "Keratoconus", "Uveitis", "Cataract"]
    assert example["task"] == "medmcqa"


def test_stratified_sample_indices_keeps_label_proportions() -> None:
    labels = ["yes"] * 60 + ["no"] * 30 + ["maybe"] * 10

    indices = stratified_sample_indices(labels, size=10, seed=0)

    picked = [labels[i] for i in indices]
    assert len(indices) == 10
    assert (picked.count("yes"), picked.count("no"), picked.count("maybe")) == (6, 3, 1)


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


def task_rows(task: str, count: int, labels: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [f"{task} {i}" for i in range(count)],
            "completion": ["x"] * count,
            "task": [task] * count,
            "label": [labels[i % len(labels)] for i in range(count)],
        }
    )


def test_build_mixture_has_the_configured_examples_per_epoch() -> None:
    mixture_config = MixtureConfig(
        epochs=2,
        medtext_repeats=2,
        pubmedqa_labeled_repeats=3,
        pubmedqa_artificial_per_class=5,
        medmcqa_per_epoch=7,
        medqa_repeats=2,
    )
    splits = DatasetDict(
        {
            "train_medtext": task_rows("medtext", 4, [""]),
            "train_pubmedqa": task_rows("pubmedqa", 2, ["yes"]),
            "train_pubmedqa_artificial": task_rows("pubmedqa_artificial", 30, ["yes", "yes", "no"]),
            "train_medmcqa": task_rows("medmcqa", 10, ["A"]),
            "train_medqa": task_rows("medqa", 3, ["B"]),
        }
    )

    mixture = build_mixture(splits, mixture_config, seed=0)

    tasks = mixture["task"]
    assert tasks.count("medtext") == 2 * 4 * 2
    assert tasks.count("pubmedqa") == 2 * 2 * 3
    assert tasks.count("pubmedqa_artificial") == 2 * (5 + 5)
    assert tasks.count("medmcqa") == 2 * 7
    assert tasks.count("medqa") == 2 * 3 * 2
    assert len(mixture) == 16 + 12 + 20 + 14 + 12


def test_save_splits_writes_one_jsonl_per_split_that_load_multitask_reads_back(tmp_path: Path) -> None:
    splits = {
        "train_medqa": task_rows("medqa", 3, ["B"]),
        "test_medqa": Dataset.from_dict(
            {"prompt": ["Qual é a dose?"], "completion": ["5 µg"], "task": ["medqa"], "label": ["A"]}
        ),
    }

    save_splits(splits, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["test_medqa.jsonl", "train_medqa.jsonl"]
    first_line = (tmp_path / "test_medqa.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line)["completion"] == "5 µg"
    assert "µg" in first_line
    loaded = load_multitask(tmp_path)
    assert set(loaded) == {"train_medqa", "test_medqa"}
    assert loaded["train_medqa"].to_list() == splits["train_medqa"].to_list()


def test_load_multitask_names_the_directory_without_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_multitask(tmp_path)


def test_to_examples_uses_the_same_string_type_for_every_source() -> None:
    large = Features({"prompt": Value("large_string"), "completion": Value("large_string")})
    medtext = Dataset.from_dict({"prompt": ["Q?"], "completion": ["A."]}).cast(large)

    examples = to_examples(medtext, medtext_example, num_proc=1)

    assert examples.features == EXAMPLE_FEATURES
    assert len(concatenate_datasets([examples, examples])) == 2


@pytest.mark.parametrize(
    ("explanation", "option", "expected"),
    [
        # Answer restated before the explanation: dropped, explanation kept
        (
            "Ans. is 'c' i.e., Mite. Scrub typhus is transmitted by trombiculid mites.",
            "Mite",
            "Scrub typhus is transmitted by trombiculid mites.",
        ),
        (
            "Ans. D: Extremely painful Depth of burn injury decides severity",
            "Extremely painful",
            "Depth of burn injury decides severity",
        ),
        (
            "Answer is A (PPI) PPIs are the treatment of choice for Zollinger-Ellison syndrome.",
            "PPI",
            "PPIs are the treatment of choice for Zollinger-Ellison syndrome.",
        ),
        # Bullet left over from the source formatting
        (
            "Ans. is 'c' i.e.. Arthritis o Lyme disease is the most common tick-borne disease.",
            "Arthritis",
            "Lyme disease is the most common tick-borne disease.",
        ),
        # Textbook reference: removed to the end
        (
            "Acute hemolysis is triggered by fava beans, infections and drugs. Ref: Harrison, 18th edition, Page 878.",
            "G6PD deficiency",
            "Acute hemolysis is triggered by fava beans, infections and drugs.",
        ),
        (
            "Secretory proteins are synthesized in the rough ER. Reference: Harper; 30th edition",
            "Rough ER",
            "Secretory proteins are synthesized in the rough ER.",
        ),
        # Reference in parentheses mid-text: only the parentheses are removed
        (
            "a. Aldolase B(Ref: Nelson's 20/e p 720-728)In the given scenario the child ingested fructose.",
            "Aldolase B",
            "In the given scenario the child ingested fructose.",
        ),
        # The explanation that follows a citation is kept
        (
            "Ref: Pal GK. Textbook of medical physiology. 2011: 4: 32.Explanation: Permeability depends on lipid solubility.",
            "Lipid solubility",
            "Permeability depends on lipid solubility.",
        ),
        (
            "Glomerular injury shows RBC casts in urine microscopy. Ref Harrison20th edition pg 292",
            "RBC casts",
            "Glomerular injury shows RBC casts in urine microscopy.",
        ),
        # "reference" as an ordinary word is not a citation
        (
            "Bleeding time is the time a skin puncture takes to stop bleeding. Normal reference value is 2-9 minutes.",
            "Bleeding time",
            "Bleeding time is the time a skin puncture takes to stop bleeding. Normal reference value is 2-9 minutes.",
        ),
        # Explanations without either pattern are unchanged
        (
            "In positively skewed distributions the mean is greater than the median.",
            "Mean",
            "In positively skewed distributions the mean is greater than the median.",
        ),
    ],
)
def test_clean_medmcqa_explanation_removes_restated_answers_and_references(
    explanation: str, option: str, expected: str
) -> None:
    assert clean_medmcqa_explanation(explanation, option) == expected


@pytest.mark.parametrize(
    ("explanation", "option"),
    [
        ("Ans. C i.e. Mite", "Mite"),
        ("Ans. is 'a' i.e., Piecemeal necrosis", "Piecemeal necrosis"),
        ("Ans. All of the above", "All of the above"),
        ("Ans. (d) Ca2+Ref : Harrison's Internal medicine 18/e, chapter 387", "Ca2+"),
        (None, "Mite"),
        ("None", "Mite"),
    ],
)
def test_clean_medmcqa_explanation_returns_none_when_nothing_is_left(
    explanation: str | None, option: str
) -> None:
    assert clean_medmcqa_explanation(explanation, option) is None


def test_medmcqa_example_uses_the_cleaned_explanation() -> None:
    row = {
        "question": "Scrub typhus is transmitted by",
        "opa": "Louse",
        "opb": "Tick",
        "opc": "Mite",
        "opd": "Milk",
        "cop": 2,
        "exp": "Ans. C i.e. Mite. It is spread by larval trombiculid mites. Ref: Park 23rd ed",
    }

    example = medmcqa_example(row)

    assert example["completion"] == "Mite\n\nExplanation: It is spread by larval trombiculid mites."


def test_medqa_example_asks_the_vignette_open_ended() -> None:
    row = {
        "question": "A pregnant woman has burning on urination. Which of the following is the best treatment for this patient?",
        "options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin"},
        "answer_idx": "D",
    }

    example = medqa_example(row)

    assert example["prompt"] == format_prompt(
        "A pregnant woman has burning on urination. What is the best treatment for this patient?"
    )
    assert example["completion"] == "Nitrofurantoin"
    assert example["task"] == "medqa"
    assert example["label"] == "D"
    assert json.loads(example["options"])[3] == "Nitrofurantoin"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which of the following is the most likely diagnosis?", "What is the most likely diagnosis?"),
        ("This drug will also prevent which of the following conditions?", "This drug will also prevent what conditions?"),
        ("Thinning of cornea occurs in", "Thinning of cornea occurs in"),
    ],
)
def test_open_question_drops_the_reference_to_options(question: str, expected: str) -> None:
    assert open_question(question) == expected


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("All are features of nephrotic syndrome except", "Hematuria"),
        ("What is true about scrub typhus?", "Caused by a mite"),
        ("Drug not used in malaria is", "Penicillin"),
        ("Most common site of carcinoid is", "All of the above"),
        ("Features of Down syndrome include", "Both a and b"),
        ("The following are seen in scurvy", "Bleeding gums"),
        ("A 30-year-old man does not smoke. What is the least likely cause?", "Asthma"),
    ],
)
def test_needs_options_finds_questions_that_only_make_sense_with_the_options(question: str, answer: str) -> None:
    assert needs_options(question, answer) is True


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Thinning of cornea occurs in", "Keratoconus"),
        ("A 30-year-old man does not smoke. He has chest pain. What is the most likely diagnosis?", "Angina"),
        ("Treatment of alcohol dependence includes?", "Chlordiazepoxide"),
    ],
)
def test_needs_options_keeps_questions_with_a_standalone_answer(question: str, answer: str) -> None:
    assert needs_options(question, answer) is False


def medmcqa_rows(questions: list[str]) -> Dataset:
    count = len(questions)
    return Dataset.from_dict(
        {
            "question": questions,
            "opa": ["Glaucoma"] * count,
            "opb": ["Keratoconus"] * count,
            "opc": ["Uveitis"] * count,
            "opd": ["Cataract"] * count,
            "cop": [1] * count,
            "exp": [None] * count,
        }
    )


def medqa_rows(count: int) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [f"A patient {i}. What is the diagnosis?" for i in range(count)],
            "options": [{"A": "Flu", "B": "Cold", "C": "Asthma", "D": "Gout"}] * count,
            "answer_idx": ["C"] * count,
        }
    )


def test_build_examples_converts_each_source_and_keeps_the_official_tests_apart() -> None:
    raw = {
        "medtext": DatasetDict(
            {"train": Dataset.from_dict({"Prompt": ["Q1", "Q1", " Q2 "], "Completion": ["A1", "A1 again", "A2"]})}
        ),
        "pubmedqa_labeled": DatasetDict({"train": pubmedqa_rows(4, ["yes", "no"])}),
        "pubmedqa_artificial": DatasetDict({"train": pubmedqa_rows(6, ["yes"])}),
        "medmcqa": DatasetDict(
            {
                "train": medmcqa_rows(["Thinning of cornea occurs in", "All of the following are true except"]),
                "validation": medmcqa_rows(["Corneal ectasia is called"]),
            }
        ),
        "medqa": DatasetDict({"train": medqa_rows(3), "test": medqa_rows(2)}),
    }

    examples, stats = build_examples(raw, test_pmids={"1"}, num_proc=1)

    assert {name: len(split) for name, split in examples.items()} == {
        "medtext": 2,
        "pubmedqa_train": 3,
        "pubmedqa_test": 1,
        "pubmedqa_artificial": 2,
        "medmcqa_train": 1,
        "medmcqa_test": 1,
        "medqa_train": 3,
        "medqa_test": 2,
    }
    # pubid 1 is the official test question (decision "no")
    assert examples["pubmedqa_test"]["label"] == ["no"]
    assert examples["medtext"]["prompt"] == [format_prompt("Q1"), format_prompt("Q2")]
    assert examples["medqa_train"]["completion"] == ["Asthma"] * 3
    assert stats["dropped_needs_options"]["medmcqa_train"] == 1
    assert stats["pubmedqa_artificial_overlap_removed"] == 4
    assert all(split.features == EXAMPLE_FEATURES for split in examples.values())


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
