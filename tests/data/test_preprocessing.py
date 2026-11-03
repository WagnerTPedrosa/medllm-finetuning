import json

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets

from medllm.data.preprocessing import (
    EXAMPLE_FEATURES,
    preprocess,
    clean_examples,
    clean_medmcqa_explanation,
    format_open_completion,
    format_prompt,
    format_pubmedqa_completion,
    format_pubmedqa_prompt,
    medmcqa_example,
    medqa_example,
    medtext_example,
    needs_options,
    open_question,
    pubmedqa_example,
    to_examples,
)


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


def test_format_prompt_wraps_question_in_template() -> None:
    assert format_prompt("Fever?") == "Question:\nFever?\n\nAnswer:\n"


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


def test_clean_examples_strips_drops_empty_and_duplicates() -> None:
    raw = pd.DataFrame(
        {
            "Prompt": ["  a  ", "a", "", None, "b"],
            "Completion": ["x", "y", "z", "w", "  "],
        }
    )

    cleaned, counts = clean_examples(raw)

    assert cleaned.to_dict("records") == [{"prompt": "a", "completion": "x"}]
    assert counts == {"raw": 5, "after_empty_removal": 2, "after_dedup": 1}


def test_clean_examples_raises_for_missing_columns() -> None:
    with pytest.raises(ValueError, match="Completion"):
        clean_examples(pd.DataFrame({"Prompt": ["a"]}))


def test_preprocess_converts_each_source_and_keeps_the_official_tests_apart() -> None:
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

    examples, stats = preprocess(raw, test_pmids={"1"}, num_proc=1)

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
