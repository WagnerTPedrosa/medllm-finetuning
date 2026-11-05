import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from medllm.chat import chat_prompt, prompt_text
from medllm.evaluation.generation import generate_completion, generate_response

TOKENIZER_ID = "Qwen/Qwen3-1.7B-Base"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=len(tokenizer))).eval()


class EndTurnModel(torch.nn.Module):
    """Fake LM that writes one token, then ends its turn."""

    def __init__(self, first_token: int, end_token: int, vocab_size: int) -> None:
        super().__init__()
        self.first_token = first_token
        self.end_token = end_token
        self.vocab_size = vocab_size
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs: object) -> torch.Tensor:
        new = torch.tensor([[self.first_token, self.end_token, self.first_token]])
        return torch.cat([input_ids, new], dim=1)


def test_generate_completion_uses_the_prompt_as_given(tiny_model, tokenizer) -> None:
    prompt = prompt_text(tokenizer, chat_prompt("Be accurate.", "Does X help?"))

    completion = generate_completion(tiny_model, tokenizer, prompt, torch.device("cpu"), max_new_tokens=5)

    assert isinstance(completion, str)
    assert prompt not in completion


def test_generate_completion_stops_at_the_end_of_the_turn(tokenizer) -> None:
    word = tokenizer.encode("Rest", add_special_tokens=False)[0]
    model = EndTurnModel(word, tokenizer.convert_tokens_to_ids("<|im_end|>"), len(tokenizer))

    completion = generate_completion(model, tokenizer, "prompt", torch.device("cpu"), max_new_tokens=5)  # type: ignore[arg-type]

    assert completion == "Rest"


def test_generate_response_asks_with_the_chat_template(tiny_model, tokenizer) -> None:
    device = torch.device("cpu")

    response = generate_response(tiny_model, tokenizer, "What causes asthma?", "Be accurate.", device, max_new_tokens=5)

    prompt = prompt_text(tokenizer, chat_prompt("Be accurate.", "What causes asthma?"))
    assert response == generate_completion(tiny_model, tokenizer, prompt, device, max_new_tokens=5)
