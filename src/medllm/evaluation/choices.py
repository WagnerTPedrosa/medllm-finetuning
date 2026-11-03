"""Evaluate questions with a known set of answers by scoring each possible answer.

For every question the model scores each choice as the log-probability of its
tokens right after the prompt, and the highest-scoring choice is the prediction.
This avoids parsing generated text, which is unreliable for small models.

- Fixed choices (PubMedQA: "yes"/"no"/"maybe"): summed log-probability.
- Open questions with their own options (MedMCQA, MedQA asked without options):
  each option text is scored as the answer, divided by its length in characters
  so long options are not penalized for having more tokens (like `acc_norm` in
  lm-evaluation-harness). The model never sees the options or their letters.
"""

import json
from typing import Any

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase

SequenceItem = tuple[list[int], int]


def score_batch(
    model: PreTrainedModel,
    items: list[SequenceItem],
    pad_token_id: int,
    device: torch.device,
) -> list[float]:
    """Log-probability of the tokens after the prompt, for each (token ids, prompt length).

    Sequences are right-padded; padding never contributes to the score.
    """
    max_length: int = max(len(ids) for ids, _ in items)
    input_ids = torch.full((len(items), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_length), dtype=torch.long)
    for row, (ids, _) in enumerate(items):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention_mask[row, : len(ids)] = 1

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
        ).logits.float()
    # The logits at position t predict the token at position t + 1
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:].to(device)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu()

    scores: list[float] = []
    for row, (ids, prompt_length) in enumerate(items):
        scores.append(token_log_probs[row, prompt_length - 1 : len(ids) - 1].sum().item())
    return scores


def score_items(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: list[SequenceItem],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    """Log-probability of the answer tokens of every (token ids, prompt length), in batches."""
    scores: list[float] = []
    for start in range(0, len(items), batch_size):
        scores.extend(score_batch(model, items[start : start + batch_size], tokenizer.eos_token_id, device))
    return scores


def choice_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    choices: list[str],
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    """Score every choice after every prompt.

    Args:
        model: Causal language model (in eval mode).
        tokenizer: Tokenizer matching the model.
        prompts: Prompts ending right where the answer starts.
        choices: Possible answers, e.g. `["yes", "no", "maybe"]`.
        device: Device the model is on.
        batch_size: Sequences per forward pass.

    Returns:
        One list of choice scores (log-probabilities) per prompt.
    """
    choice_ids: list[list[int]] = [
        tokenizer.encode(choice, add_special_tokens=False) for choice in choices
    ]
    items: list[SequenceItem] = []
    for prompt in prompts:
        prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
        items.extend((prompt_ids + ids, len(prompt_ids)) for ids in choice_ids)

    flat_scores: list[float] = score_items(model, tokenizer, items, device, batch_size)
    return [
        flat_scores[start : start + len(choices)]
        for start in range(0, len(flat_scores), len(choices))
    ]


def option_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    options: list[list[str]],
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    """Score each prompt's own options as answers, per character of the option.

    Args:
        model: Causal language model (in eval mode).
        tokenizer: Tokenizer matching the model.
        prompts: Open questions, ending right where the answer starts.
        options: The option texts of each prompt.
        device: Device the model is on.
        batch_size: Sequences per forward pass.

    Returns:
        One list of scores (log-probability / characters) per prompt.
    """
    items: list[SequenceItem] = []
    lengths: list[int] = []
    for prompt, prompt_options in zip(prompts, options):
        prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
        for option in prompt_options:
            items.append((prompt_ids + tokenizer.encode(option, add_special_tokens=False), len(prompt_ids)))
            lengths.append(max(1, len(option)))

    flat_scores: list[float] = score_items(model, tokenizer, items, device, batch_size)
    normalized: list[float] = [score / length for score, length in zip(flat_scores, lengths)]
    grouped: list[list[float]] = []
    start = 0
    for prompt_options in options:
        grouped.append(normalized[start : start + len(prompt_options)])
        start += len(prompt_options)
    return grouped


def classification_metrics(
    gold: list[str], predicted: list[str], labels: list[str]
) -> dict[str, float]:
    """Accuracy and macro F1 (every label weighs the same, whatever its frequency)."""
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)),
    }


def evaluate_choice_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    choices: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Predict the most likely choice for each example and score the predictions.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer matching the model.
        dataset: Examples with `prompt` and gold `label` (one of `choices`).
        choices: Possible answers.
        device: Device the model is on.
        batch_size: Sequences per forward pass.

    Returns:
        `accuracy`, `macro_f1` and the list of `predictions`.
    """
    model.eval()
    scores: list[list[float]] = choice_scores(
        model, tokenizer, dataset["prompt"], choices, device, batch_size
    )
    predictions: list[str] = [choices[row.index(max(row))] for row in scores]
    return {**classification_metrics(dataset["label"], predictions, choices), "predictions": predictions}


def evaluate_open_choice_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    letters: str = "ABCD",
) -> dict[str, Any]:
    """Predict the best of each open question's own options and score the predictions.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer matching the model.
        dataset: Examples with an open `prompt`, `options` (JSON list of option
            texts) and the gold `label` (letter of the correct option).
        device: Device the model is on.
        batch_size: Sequences per forward pass.
        letters: Letters of the options, in order.

    Returns:
        `accuracy`, `macro_f1` (over the option letters) and the predicted letters.
    """
    model.eval()
    options: list[list[str]] = [json.loads(value) for value in dataset["options"]]
    scores = option_scores(model, tokenizer, dataset["prompt"], options, device, batch_size)
    predictions: list[str] = [letters[row.index(max(row))] for row in scores]
    return {
        **classification_metrics(dataset["label"], predictions, list(letters[: len(options[0])])),
        "predictions": predictions,
    }
