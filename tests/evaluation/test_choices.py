import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from medllm.evaluation.choices import (
    choice_scores,
    classification_metrics,
    evaluate_choice_task,
    evaluate_open_choice_task,
    option_scores,
)

TOKENIZER_ID = "openai-community/gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


class PreferTokenModel(torch.nn.Module):
    """Fake LM that always predicts one token id with near certainty."""

    def __init__(self, token_id: int, vocab_size: int) -> None:
        super().__init__()
        self.token_id = token_id
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        logits = torch.zeros(*input_ids.shape, self.vocab_size)
        logits[..., self.token_id] = 20.0
        return SimpleNamespace(logits=logits)


def test_choice_scores_match_manual_log_probabilities(tokenizer) -> None:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16)).eval()
    prompt, choice = "Answer:\n", "yes"

    scores = choice_scores(model, tokenizer, [prompt, prompt], ["yes", "no"], torch.device("cpu"), batch_size=3)

    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids + tokenizer.encode(choice)])
    with torch.no_grad():
        log_probs = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
    expected = sum(
        log_probs[position - 1, ids[0, position]].item()
        for position in range(len(prompt_ids), ids.shape[1])
    )
    assert len(scores) == 2 and len(scores[0]) == 2
    assert scores[0][0] == pytest.approx(expected, abs=1e-4)
    assert scores[1] == pytest.approx(scores[0], abs=1e-4)


def test_evaluate_choice_task_picks_the_most_likely_choice(tokenizer) -> None:
    no_token = tokenizer.encode("no")[0]
    model = PreferTokenModel(no_token, tokenizer.vocab_size)
    dataset = Dataset.from_dict(
        {"prompt": ["Q1\nAnswer:\n", "Q2\nAnswer:\n"], "label": ["no", "yes"]}
    )

    result = evaluate_choice_task(
        model, tokenizer, dataset, ["yes", "no", "maybe"], torch.device("cpu"), batch_size=4
    )

    assert result["predictions"] == ["no", "no"]
    assert result["accuracy"] == 0.5


def test_classification_metrics_reports_accuracy_and_macro_f1() -> None:
    metrics = classification_metrics(
        ["yes", "yes", "no", "maybe"], ["yes", "no", "no", "yes"], ["yes", "no", "maybe"]
    )

    assert metrics["accuracy"] == 0.5
    # per-class F1: yes 0.5, no 0.667, maybe 0
    assert metrics["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0.0) / 3)


def test_option_scores_are_choice_scores_divided_by_the_option_length(tokenizer) -> None:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16)).eval()
    prompt, options = "Answer:\n", ["Keratoconus", "Glaucoma"]
    device = torch.device("cpu")

    scores = option_scores(model, tokenizer, [prompt], [options], device, batch_size=2)

    raw = choice_scores(model, tokenizer, [prompt], options, device, batch_size=2)[0]
    assert scores[0] == pytest.approx([raw[0] / len(options[0]), raw[1] / len(options[1])], abs=1e-5)


def test_evaluate_open_choice_task_predicts_the_letter_of_the_best_option(tokenizer) -> None:
    no_token = tokenizer.encode("no")[0]
    model = PreferTokenModel(no_token, tokenizer.vocab_size)
    dataset = Dataset.from_dict(
        {
            "prompt": ["Q1\nAnswer:\n", "Q2\nAnswer:\n"],
            "options": [json.dumps(["yes", "no", "maybe", "never"]), json.dumps(["no", "yes", "maybe", "never"])],
            "label": ["B", "B"],
        }
    )

    result = evaluate_open_choice_task(model, tokenizer, dataset, torch.device("cpu"), batch_size=4)

    assert result["predictions"] == ["B", "A"]
    assert result["accuracy"] == 0.5
