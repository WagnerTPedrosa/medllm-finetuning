import math
from pathlib import Path

import mlflow
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from src.config import PretrainConfig, load_pretrain_config
from src.pretrain import (
    PerplexityCallback,
    build_training_arguments,
    collate_blocks,
    with_learning_rate,
)
from src.tracking import configure_tracking

PRETRAIN_CONFIG_PATH = Path("configs/gpt2_pubmed_pretrain.yaml").resolve()
EOS = 50256


def load() -> PretrainConfig:
    return load_pretrain_config(PRETRAIN_CONFIG_PATH)


def test_load_pretrain_config_reads_pretraining_section() -> None:
    training = load().pretraining

    assert training.lr_scheduler_type == "cosine"
    assert training.bf16 is True
    assert training.per_device_batch_size * training.gradient_accumulation_steps == 512


def test_collate_blocks_uses_input_ids_as_labels_including_eos() -> None:
    features = [{"input_ids": [1, 2, EOS, 3]}, {"input_ids": [EOS, 4, 5, 6]}]

    batch = collate_blocks(features)

    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].shape == (2, 4)
    # The EOS between documents is a prediction target, not padding
    assert torch.equal(batch["labels"], batch["input_ids"])


def test_build_training_arguments_follows_config(tmp_path: Path) -> None:
    config = load()

    args = build_training_arguments(config, tmp_path, max_steps=None)

    training = config.pretraining
    assert args.learning_rate == training.learning_rate
    assert args.gradient_accumulation_steps == training.gradient_accumulation_steps
    assert args.eval_steps == training.eval_steps
    assert args.max_steps == -1
    assert args.report_to == ["mlflow"]
    assert args.save_total_limit == training.save_total_limit
    assert args.torch_compile == training.torch_compile
    assert args.optim == training.optim


def test_build_training_arguments_overrides_max_steps_for_smoke_runs(tmp_path: Path) -> None:
    args = build_training_arguments(load(), tmp_path, max_steps=20)

    assert args.max_steps == 20


def test_perplexity_callback_logs_exp_of_eval_loss(tracking_store: Path) -> None:
    config = load()
    configure_tracking(config)
    state = TrainerState()
    state.global_step = 40

    with mlflow.start_run() as run:
        PerplexityCallback().on_evaluate(
            TrainingArguments(output_dir=str(tracking_store), report_to="none"),
            state,
            TrainerControl(),
            metrics={"eval_loss": 2.0},
        )

    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "eval_perplexity")
    assert [(point.step, round(point.value, 4)) for point in history] == [
        (40, round(math.exp(2.0), 4))
    ]


def test_load_pretrain_config_reads_lr_sweep_section() -> None:
    sweep = load().lr_sweep

    assert sweep.learning_rates == [5e-5, 1e-4, 3e-4]
    assert all(isinstance(rate, float) for rate in sweep.learning_rates)
    assert sweep.max_steps == 150


def test_with_learning_rate_returns_a_new_config() -> None:
    config = load()

    updated = with_learning_rate(config, 3e-4)

    assert updated.pretraining.learning_rate == 3e-4
    assert config.pretraining.learning_rate == 1e-4
    assert updated.corpus == config.corpus
