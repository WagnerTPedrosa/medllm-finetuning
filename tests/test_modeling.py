from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.modeling import (
    build_peft_config,
    load_causal_lm,
    merge_lora,
    resolve_model_source,
    set_end_of_turn,
    weights_dtype,
)
from medllm.tracking import configure_tracking


def test_load_causal_lm_loads_weights_in_float32(tmp_path: Path) -> None:
    # Checkpoints saved in bf16 (like most recent Hub models) would otherwise be
    # trained in pure bf16, which loses small optimizer updates
    GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16)).to(torch.bfloat16).save_pretrained(tmp_path)

    model = load_causal_lm(str(tmp_path))

    assert next(model.parameters()).dtype == torch.float32


def test_load_causal_lm_can_load_bf16_weights_for_lora(tmp_path: Path) -> None:
    GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16)).save_pretrained(tmp_path)

    model = load_causal_lm(str(tmp_path), dtype=weights_dtype("lora"))

    assert next(model.parameters()).dtype == torch.bfloat16


def test_weights_dtype_is_bf16_for_lora_and_float32_for_full_fine_tuning() -> None:
    # LoRA trains small float32 adapters on frozen weights, so the frozen
    # weights can stay in bf16; full fine-tuning updates them and needs float32
    assert weights_dtype("lora") == torch.bfloat16
    assert weights_dtype("full") == torch.float32


def test_set_end_of_turn_makes_generation_stop_after_the_answer() -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")

    set_end_of_turn(model, tokenizer)

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert model.generation_config.eos_token_id == [im_end, tokenizer.eos_token_id]
    assert model.generation_config.pad_token_id == tokenizer.pad_token_id


def test_resolve_model_source_keeps_hub_ids_and_local_paths() -> None:
    assert resolve_model_source("openai-community/gpt2") == "openai-community/gpt2"
    assert resolve_model_source("checkpoints/gpt2-medtext-best") == "checkpoints/gpt2-medtext-best"


def test_resolve_model_source_downloads_logged_model_weights(
    config: MultitaskConfig, tracking_store: Path
) -> None:
    configure_tracking(config)
    packaged = tracking_store / "packaged"
    (packaged / "model").mkdir(parents=True)
    (packaged / "model" / "config.json").write_text('{"model_type": "gpt2"}')
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(packaged), "packaged")

    local_dir = Path(resolve_model_source(f"runs:/{run.info.run_id}/packaged"))

    assert local_dir.name == "model"
    assert (local_dir / "config.json").exists()


def load() -> MultitaskConfig:
    return load_multitask_config()


def test_build_peft_config_follows_the_lora_settings() -> None:
    config = load()

    peft_config = build_peft_config(config)

    assert isinstance(peft_config, LoraConfig)
    assert peft_config.r == config.finetune.lora.r
    assert peft_config.lora_alpha == config.finetune.lora.alpha
    assert set(peft_config.target_modules) == set(config.finetune.lora.target_modules)
    # The chat tokens are untrained in the base model: the (tied) embeddings are trained too
    assert peft_config.modules_to_save == ["embed_tokens"]
    assert peft_config.ensure_weight_tying


def test_build_peft_config_is_none_for_full_fine_tuning() -> None:
    assert build_peft_config(load_multitask_config(["finetune.method=full"])) is None


def test_merged_lora_model_matches_the_adapter_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16)).eval()
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], init_lora_weights=False)
    adapted = get_peft_model(base, lora)
    inputs = torch.tensor([[1, 2, 3]])
    expected = adapted(inputs).logits

    merged = merge_lora(adapted)

    assert not isinstance(merged, PeftModel)
    assert torch.allclose(merged(inputs).logits, expected, atol=1e-5)
