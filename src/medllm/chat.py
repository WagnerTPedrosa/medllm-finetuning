"""Chat format: the messages of an example, the text the model continues, and where its turn ends.

The model is fine-tuned on conversations rendered with its own chat template
(ChatML for Qwen), so it can be served as a chat model (vLLM, OpenAI-compatible
APIs, the dashboard) with the same format it was trained on.
"""

from transformers import PreTrainedTokenizerBase

Message = dict[str, str]

# ChatML closes every turn with this token: the model writes it after its answer
END_OF_TURN = "<|im_end|>"


def chat_prompt(system_prompt: str, user_content: str) -> list[Message]:
    """The system prompt and the user's question, as chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def assistant_reply(content: str) -> list[Message]:
    """The answer the model should write, as one assistant message."""
    return [{"role": "assistant", "content": content}]


def message_text(messages: list[Message]) -> str:
    """Content of the last message: the user's question of a prompt, or the answer of a completion."""
    return messages[-1]["content"]


def prompt_text(tokenizer: PreTrainedTokenizerBase, messages: list[Message]) -> str:
    """The text the model continues to answer: the chat template up to the assistant's answer.

    Thinking is off, so Qwen3's template ends with an empty `<think></think>`
    block. Training renders that same empty block before every answer, so the
    answer is what comes next in both cases.

    Args:
        tokenizer: Tokenizer with a chat template.
        messages: System and user messages.

    Returns:
        The prompt, special tokens included as text.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Token ids that end the model's turn: END_OF_TURN (when in the vocabulary), then EOS."""
    ids: list[int] = []
    end_of_turn: int | None = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if end_of_turn is not None and end_of_turn != tokenizer.unk_token_id:
        ids.append(end_of_turn)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids
