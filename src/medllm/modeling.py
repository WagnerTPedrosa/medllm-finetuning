"""Device selection and model loading."""

import gc
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from medllm.chat import stop_token_ids
from medllm.config import MultitaskConfig

MLFLOW_URI_PREFIXES = ("models:/", "runs:/")
DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def resolve_model_source(model_id: str) -> str:
    """Turn a model reference into something `from_pretrained` can load.

    Hub ids and local directories are returned unchanged. MLflow URIs
    (`models:/name@alias`, `models:/name/3`, `runs:/<run_id>/model`) are
    downloaded, and the directory holding the weights and tokenizer is
    returned (MLflow stores them in a `model/` subdirectory).

    Args:
        model_id: Hub id, local path or MLflow model URI.

    Returns:
        A Hub id or a local directory.
    """
    if not model_id.startswith(MLFLOW_URI_PREFIXES):
        return model_id

    local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_id))
    weights_dir = local_dir / "model"
    if (weights_dir / "config.json").exists():
        return str(weights_dir)
    return str(local_dir)


def get_device() -> torch.device:
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def weights_dtype(method: str) -> torch.dtype:
    """Dtype of the base model's weights for a fine-tuning method.

    LoRA keeps the base weights frozen and trains small adapters (in float32),
    so the frozen weights can stay in bf16: half the memory, faster steps. Full
    fine-tuning updates every weight and keeps them in float32, so small
    updates are not lost (bf16 is used only for the computations).

    Args:
        method: `lora` or `full`.

    Returns:
        `torch.bfloat16` for LoRA, `torch.float32` otherwise.
    """
    return torch.bfloat16 if method == "lora" else torch.float32


def parse_dtype(name: str) -> torch.dtype:
    """The torch dtype named in the configuration (`float32` or `bfloat16`).

    Raises:
        ValueError: If the name is not one of `DTYPES`.
    """
    if name not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}, got {name!r}")
    return DTYPES[name]


def load_causal_lm(model_id: str, dtype: torch.dtype = torch.float32) -> PreTrainedModel:
    """Load a causal language model with weights in `dtype` (float32 by default).

    Transformers loads weights in the dtype they were saved in (bf16 for most
    recent Hub models) unless told otherwise, so the dtype is always explicit.

    Args:
        model_id: Hub id, local checkpoint directory or MLflow model URI.
        dtype: Dtype of the weights (see `weights_dtype`).

    Returns:
        The model, on the CPU.
    """
    return AutoModelForCausalLM.from_pretrained(resolve_model_source(model_id), dtype=dtype)


def set_end_of_turn(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Make generation stop where the model's chat turn ends, and save that with the model.

    The fine-tuned model ends every answer with the chat template's end-of-turn
    token, not with EOS; vLLM and `generate` read the stop tokens from the
    model's generation config.

    Args:
        model: Fine-tuned model (its generation config is changed in place).
        tokenizer: Its tokenizer.
    """
    model.generation_config.eos_token_id = stop_token_ids(tokenizer)
    model.generation_config.pad_token_id = tokenizer.pad_token_id


def load_model(model_id: str) -> tuple[PreTrainedModel, torch.device]:
    """Load a causal language model onto the best available device.

    Args:
        model_id: Hub id, local checkpoint directory or MLflow model URI.

    Returns:
        The model and the device it was moved to.
    """
    device: torch.device = get_device()

    model: PreTrainedModel = load_causal_lm(model_id)
    model.config.use_cache = False
    model.to(device)

    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    return model, device


def free_gpu_memory() -> None:
    """Release cached GPU memory between runs in the same process."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_label(model_id: str) -> str:
    """Short name for directories and run names (`gpt2`, `gpt2-pubmed-dapt@champion`)."""
    return model_id.rstrip("/").split("/")[-1]


def build_peft_config(config: MultitaskConfig) -> LoraConfig | None:
    """LoRA adapters from `finetune.lora` when `finetune.method` is `lora`, else None.

    The modules in `modules_to_save` are trained in full; tied weights (Qwen's
    embeddings and LM head) stay tied, so the model loads back as trained.
    """
    if config.finetune.method != "lora":
        return None
    lora = config.finetune.lora
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        modules_to_save=list(lora.modules_to_save),
        ensure_weight_tying=True,
        bias="none",
        task_type="CAUSAL_LM",
    )


def merge_lora(model: PreTrainedModel | PeftModel) -> PreTrainedModel:
    """Fold LoRA adapters into the base weights; a model without adapters is returned as is.

    The merged model is a plain checkpoint, so evaluation, generation and the
    Model Registry handle LoRA and fully fine-tuned models the same way.
    """
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model
