"""Score a model on the four tasks: MedText answer loss, and accuracy and macro F1 on the choice tasks.

PubMedQA is scored over its fixed decisions (yes/no/maybe); MedMCQA and MedQA,
asked open-ended, by the length-normalized probability of each option's text
(see `medllm.evaluation.choices`). Every answer is also marked right or wrong,
for the confidence intervals of the report.
"""

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from medllm.chat import prompt_text
from medllm.config import MultitaskConfig
from medllm.data.finetune_data import load_finetune_data
from medllm.data.preprocessing import PUBMEDQA_DECISIONS, SFT_COLUMNS
from medllm.evaluation.choices import evaluate_choice_task, evaluate_open_choice_task
from medllm.evaluation.loss import evaluate
from medllm.modeling import get_device, load_causal_lm, resolve_model_source

# Scored by picking the most likely answer: PubMedQA among its fixed decisions,
# the open MedMCQA and MedQA questions among each question's own options
FIXED_CHOICE_TASKS: dict[str, tuple[str, ...]] = {"pubmedqa": PUBMEDQA_DECISIONS}


OPEN_CHOICE_TASKS = ("medmcqa", "medqa")


@dataclass(frozen=True)
class TaskEvaluation:
    """Metrics of a model, and which test answers it got right (for confidence intervals)."""

    metrics: dict[str, float]
    # Per choice task, 1 or 0 for every example, in dataset order
    correct: dict[str, list[int]]


def correctness(predictions: list[str], labels: list[str]) -> list[int]:
    """1 where the prediction equals the gold label, else 0."""
    return [int(prediction == label) for prediction, label in zip(predictions, labels)]


def render_prompt(example: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    """The example's chat prompt as the text the model continues (for `Dataset.map`)."""
    return {"prompt": prompt_text(tokenizer, example["prompt"])}


def evaluate_on_tasks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    splits: DatasetDict,
    config: MultitaskConfig,
    prefix: str,
) -> TaskEvaluation:
    """Score a model on the MedText split and the choice tasks, with a prefix.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer matching the model.
        splits: Splits written by `medllm.data.finetune_data filter_length`.
        config: Multitask configuration.
        prefix: `test` (final results) or `validation` (model selection, sweeps).

    Returns:
        `<prefix>_medtext_loss`, and accuracy and macro F1 for PubMedQA, MedMCQA
        and MedQA, plus each answer marked right or wrong.
    """
    device: torch.device = next(model.parameters()).device
    medtext: Dataset = splits[f"{prefix}_medtext"].select_columns(SFT_COLUMNS)
    metrics: dict[str, float] = {f"{prefix}_medtext_loss": evaluate(model, tokenizer, medtext, config)}
    correct: dict[str, list[int]] = {}
    # Scoring computes log-probabilities over the whole vocabulary: small batches keep memory low
    batch_size: int = config.training.eval_batch_size
    for task in (*FIXED_CHOICE_TASKS, *OPEN_CHOICE_TASKS):
        dataset: Dataset = splits[f"{prefix}_{task}"].map(render_prompt, fn_kwargs={"tokenizer": tokenizer})
        if task in FIXED_CHOICE_TASKS:
            choices: list[str] = list(FIXED_CHOICE_TASKS[task])
            result = evaluate_choice_task(model, tokenizer, dataset, choices, device, batch_size=batch_size)
        else:
            result = evaluate_open_choice_task(model, tokenizer, dataset, device, batch_size=batch_size)
        metrics[f"{prefix}_{task}_accuracy"] = result["accuracy"]
        metrics[f"{prefix}_{task}_macro_f1"] = result["macro_f1"]
        correct[task] = correctness(result["predictions"], dataset["label"])
    return TaskEvaluation(metrics=metrics, correct=correct)


def evaluate_model(model_id: str, config: MultitaskConfig, prefix: str = "test") -> TaskEvaluation:
    """Load a model (Hub id, path or MLflow URI) and score it with `evaluate_on_tasks`."""
    source: str = resolve_model_source(model_id)
    model = load_causal_lm(source).to(get_device())
    tokenizer = AutoTokenizer.from_pretrained(source)
    return evaluate_on_tasks(model, tokenizer, load_finetune_data(config.finetune_data.processed_dir), config, prefix)
