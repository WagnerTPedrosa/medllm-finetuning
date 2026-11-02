"""Device selection and model loading."""

from pathlib import Path

import mlflow
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

MLFLOW_URI_PREFIXES = ("models:/", "runs:/")


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


def load_model(model_id: str) -> tuple[PreTrainedModel, torch.device]:
    """Load a causal language model onto the best available device.

    Args:
        model_id: Hub id, local checkpoint directory or MLflow model URI.

    Returns:
        The model and the device it was moved to.
    """
    device: torch.device = get_device()

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        resolve_model_source(model_id)
    )
    model.config.use_cache = False
    model.to(device)

    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    return model, device
