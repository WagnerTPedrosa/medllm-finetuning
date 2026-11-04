import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from medllm.chat import (
    END_OF_TURN,
    assistant_reply,
    chat_prompt,
    message_text,
    prompt_text,
    stop_token_ids,
)

QWEN_ID = "Qwen/Qwen3-1.7B-Base"


@pytest.fixture(scope="module")
def qwen_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(QWEN_ID)


def test_chat_prompt_puts_the_system_prompt_before_the_question() -> None:
    assert chat_prompt("Be accurate.", "Fever?") == [
        {"role": "system", "content": "Be accurate."},
        {"role": "user", "content": "Fever?"},
    ]


def test_assistant_reply_is_one_assistant_message() -> None:
    assert assistant_reply("Rest.") == [{"role": "assistant", "content": "Rest."}]


def test_message_text_is_the_question_of_a_prompt_and_the_answer_of_a_completion() -> None:
    assert message_text(chat_prompt("Be accurate.", "Fever?")) == "Fever?"
    assert message_text(assistant_reply("Rest.")) == "Rest."


def test_prompt_text_ends_where_the_fine_tuned_model_starts_its_answer(qwen_tokenizer) -> None:
    messages = chat_prompt("Be accurate.", "Fever?")

    text = prompt_text(qwen_tokenizer, messages)
    full = qwen_tokenizer.apply_chat_template(messages + assistant_reply("Rest."), tokenize=False)

    # Training renders the answer after an empty think block; the prompt for
    # scoring and generation includes that block, so the answer comes next
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert full == text + "Rest." + END_OF_TURN + "\n"


def test_stop_token_ids_end_generation_at_the_end_of_the_turn(qwen_tokenizer) -> None:
    ids = stop_token_ids(qwen_tokenizer)

    assert ids[0] == qwen_tokenizer.convert_tokens_to_ids(END_OF_TURN)
    assert qwen_tokenizer.eos_token_id in ids
