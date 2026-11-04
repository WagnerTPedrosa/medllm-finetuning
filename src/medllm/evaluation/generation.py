"""Answer generation with the chat template."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from medllm.chat import chat_prompt, prompt_text, stop_token_ids


def generate_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Continue an already formatted prompt with greedy decoding, until the model's turn ends.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer matching the model.
        prompt: Full prompt, chat template included (see `medllm.chat.prompt_text`).
        device: Device the model is on.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The generated text, without the prompt and anything after the end of the turn.
    """
    model.eval()
    stop_ids: list[int] = stop_token_ids(tokenizer)

    prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[-1],
            eos_token_id=stop_ids,
        )

    generated_ids: list[int] = outputs[0, input_ids.shape[1] :].tolist()
    for position, token_id in enumerate(generated_ids):
        if token_id in stop_ids:
            generated_ids = generated_ids[:position]
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    system_prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Answer a raw question as the chat model, with greedy decoding.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer matching the model, with a chat template.
        question: The user's question.
        system_prompt: System message (the one the model was trained with).
        device: Device the model is on.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The generated answer, without the prompt and anything after the end of the turn.
    """
    prompt: str = prompt_text(tokenizer, chat_prompt(system_prompt, question))
    return generate_completion(model, tokenizer, prompt, device, max_new_tokens)
