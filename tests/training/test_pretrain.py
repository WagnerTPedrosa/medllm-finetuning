import math
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from medllm.config import PretrainConfig, load_pretrain_config
from medllm.tracking import configure_tracking
from medllm.training.pretrain import (
    PerplexityCallback,
    build_training_arguments,
    collate_blocks,
    token_budget_steps,
    with_learning_rate,
)

EOS = 50256


def load() -> PretrainConfig:
    return load_pretrain_config()


def test_load_pretrain_config_reads_pretraining_section() -> None:
    training = load().pretraining

    assert training.lr_scheduler_type == "cosine"
    assert training.bf16 is True
    config = load()
    tokens_per_step = training.per_device_batch_size * training.gradient_accumulation_steps * config.model.max_seq_len
    assert tokens_per_step == 524_288  # about 0.5M tokens per optimizer step


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
    assert args.gradient_checkpointing == training.gradient_checkpointing


def test_build_training_arguments_evaluates_the_base_model_before_training(tmp_path: Path) -> None:
    args = build_training_arguments(load(), tmp_path, max_steps=None)

    assert args.eval_on_start is True


def test_build_training_arguments_overrides_max_steps_for_smoke_runs(tmp_path: Path) -> None:
    args = build_training_arguments(load(), tmp_path, max_steps=20)

    assert args.max_steps == 20


def with_max_tokens(config: PretrainConfig, max_tokens: int | None) -> PretrainConfig:
    return replace(config, pretraining=replace(config.pretraining, max_tokens=max_tokens))


def test_load_pretrain_config_trains_on_the_whole_corpus_by_default() -> None:
    assert load().pretraining.max_tokens is None


def test_token_budget_steps_is_none_without_a_budget() -> None:
    assert token_budget_steps(with_max_tokens(load(), None)) is None


def test_token_budget_steps_divides_the_budget_by_the_tokens_per_step() -> None:
    config = with_max_tokens(load(), 1_500_000_000)

    # 1.5B / 524,288 tokens per step = 2861.02 steps: whole steps only
    assert token_budget_steps(config) == 2861


def test_token_budget_steps_rejects_a_budget_below_one_step() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        token_budget_steps(with_max_tokens(load(), 1000))


def test_build_training_arguments_stops_at_the_token_budget(tmp_path: Path) -> None:
    config = with_max_tokens(load(), 1_500_000_000)

    args = build_training_arguments(config, tmp_path, max_steps=None)

    assert args.max_steps == 2861
    # A full run keeps its evaluation and checkpoint intervals
    assert args.eval_steps == config.pretraining.eval_steps
    assert args.save_steps == config.pretraining.save_steps


def test_build_training_arguments_smoke_steps_override_the_token_budget(tmp_path: Path) -> None:
    args = build_training_arguments(with_max_tokens(load(), 1_500_000_000), tmp_path, max_steps=20)

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

    assert sweep.learning_rates == [1e-5, 3e-5, 1e-4]
    assert all(isinstance(rate, float) for rate in sweep.learning_rates)
    assert sweep.max_steps == 150


def test_with_learning_rate_returns_a_new_config() -> None:
    config = load()

    updated = with_learning_rate(config, 1e-4)

    assert updated.pretraining.learning_rate == 1e-4
    assert config.pretraining.learning_rate == 3e-5
    assert updated is not config
    assert updated.corpus == config.corpus
