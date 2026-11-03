from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.modeling import load_causal_lm


def test_load_causal_lm_loads_weights_in_float32(tmp_path: Path) -> None:
    # Checkpoints saved in bf16 (like most recent Hub models) would otherwise be
    # trained in pure bf16, which loses small optimizer updates
    GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16)).to(torch.bfloat16).save_pretrained(tmp_path)

    model = load_causal_lm(str(tmp_path))

    assert next(model.parameters()).dtype == torch.float32
