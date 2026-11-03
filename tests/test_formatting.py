from datasets import Dataset

from src.formatting import format_prompt, to_sft_format


def test_format_prompt_wraps_question_in_template() -> None:
    assert format_prompt("Fever?") == "Question:\nFever?\n\nAnswer:\n"


def test_to_sft_format_formats_prompt_and_keeps_completion() -> None:
    dataset = Dataset.from_dict(
        {"prompt": ["Fever?"], "completion": ["Viral infection."], "n_tokens": [12]}
    )

    formatted: Dataset = to_sft_format(dataset)

    assert formatted.column_names == ["prompt", "completion"]
    assert formatted[0]["prompt"] == "Question:\nFever?\n\nAnswer:\n"
    assert formatted[0]["completion"] == "Viral infection."


def test_to_sft_format_does_not_modify_input() -> None:
    dataset = Dataset.from_dict({"prompt": ["Fever?"], "completion": ["Viral."]})

    to_sft_format(dataset)

    assert dataset[0]["prompt"] == "Fever?"
