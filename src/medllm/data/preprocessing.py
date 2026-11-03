"""Preprocessing: every raw dataset as prompt-completion examples in one shared format.

MedMCQA and MedQA questions are asked open-ended, without their options: the
completion is the text of the correct option (with the cleaned explanation for
MedMCQA). Questions that only make sense with the options ("All of the
following except", "Which is true about") are left out.

Every example has `prompt`, `completion`, `task`, `label` (the gold decision for
PubMedQA, the letter of the correct option for MedMCQA and MedQA, empty for
MedText) and `options` (JSON list of the option texts, used to score the open
questions as multiple choice; empty for the other tasks).
"""

import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value

PROMPT_TEMPLATE = "Question:\n{question}\n\nAnswer:\n"


EXAMPLE_COLUMNS = ["prompt", "completion", "task", "label", "options"]


# One schema for every source, so splits from different datasets can be concatenated
EXAMPLE_FEATURES = Features({column: Value("string") for column in EXAMPLE_COLUMNS})


OPTION_LETTERS = "ABCD"


PUBMEDQA_DECISIONS = ("yes", "no", "maybe")


SFT_COLUMNS = ["prompt", "completion"]


RAW_COLUMNS = ["Prompt", "Completion"]


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


def format_prompt(question: str) -> str:
    """Wrap a question in the Question/Answer template the model is trained on.

    Args:
        question: Raw question text.

    Returns:
        The prompt the model receives before generating the answer.
    """
    return PROMPT_TEMPLATE.format(question=question)


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


def outside_pmids(pmids: list[int], excluded: set[str]) -> list[bool]:
    """Keep mask for rows whose PMID is not in `excluded` (for `Dataset.filter`)."""
    return [str(pmid) not in excluded for pmid in pmids]


def preprocess(
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
