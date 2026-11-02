"""Prompt template shared by training, evaluation and inference."""

from datasets import Dataset

PROMPT_TEMPLATE = "Question:\n{question}\n\nAnswer:\n"


def format_prompt(question: str) -> str:
    """Wrap a question in the Question/Answer template the model is trained on.

    Args:
        question: Raw question text.

    Returns:
        The prompt the model receives before generating the answer.
    """
    return PROMPT_TEMPLATE.format(question=question)


def add_prompt_template(example: dict[str, str]) -> dict[str, str]:
    """Return the example's prompt wrapped in the template (for `Dataset.map`).

    Args:
        example: Row with a raw `prompt`.

    Returns:
        Mapping with the formatted `prompt`.
    """
    return {"prompt": format_prompt(example["prompt"])}


def to_sft_format(dataset: Dataset) -> Dataset:
    """Build the prompt-completion dataset that SFTTrainer expects.

    SFTTrainer appends the EOS token to each completion and computes the loss
    only on the completion.

    Args:
        dataset: Split with `prompt` and `completion` columns (others are dropped).

    Returns:
        A new dataset with the templated `prompt` and the original `completion`.
    """
    dataset = dataset.select_columns(["prompt", "completion"])
    return dataset.map(add_prompt_template)
