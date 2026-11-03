import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from medllm.data.preprocessing import format_prompt
from medllm.evaluation.generation import generate_completion, generate_response

TOKENIZER_ID = "openai-community/gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16)).eval()


def test_generate_completion_uses_the_prompt_as_given(tiny_model, tokenizer) -> None:
    prompt = "Context:\nSome text.\n\nQuestion:\nDoes X help?\n\nAnswer (yes, no or maybe):\n"

    completion = generate_completion(tiny_model, tokenizer, prompt, torch.device("cpu"), max_new_tokens=5)

    assert isinstance(completion, str)
    assert prompt not in completion
    assert len(tokenizer.encode(completion)) <= 5 + 1  # decoding and re-encoding can merge or split a token


def test_generate_response_adds_the_question_template(tiny_model, tokenizer) -> None:
    question = "What causes asthma?"
    device = torch.device("cpu")

    response = generate_response(tiny_model, tokenizer, question, device, max_new_tokens=5)

    assert response == generate_completion(tiny_model, tokenizer, format_prompt(question), device, max_new_tokens=5)
