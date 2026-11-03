"""Answer generation."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from medllm.data.preprocessing import format_prompt


def generate_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Continue an already formatted prompt with greedy decoding.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer matching the model.
        prompt: Full prompt, template included (e.g. a multitask `prompt`).
        device: Device the model is on.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The generated text, without the prompt and anything after EOS.
    """
    model.eval()

    prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids: list[int] = outputs[0, input_ids.shape[1] :].tolist()
    if tokenizer.eos_token_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(tokenizer.eos_token_id)]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Answer a raw question with greedy decoding.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer matching the model.
        question: Raw question (the Question/Answer template is added here).
        device: Device the model is on.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The generated answer, without the prompt and anything after EOS.
    """
    return generate_completion(model, tokenizer, format_prompt(question), device, max_new_tokens)
