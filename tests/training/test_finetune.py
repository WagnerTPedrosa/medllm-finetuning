from pathlib import Path

import pytest

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.training.finetune import (
    base_model_id,
    build_multitask_sft_config,
    epoch_metrics,
    finetune_stage,
    steps_per_epoch,
    summarize_seeds,
)


def load() -> MultitaskConfig:
    return load_multitask_config()


def test_steps_per_epoch_rounds_up() -> None:
    assert steps_per_epoch(examples_per_epoch=100, batch_size=8) == 13


def test_build_multitask_sft_config_evaluates_once_per_mixture_epoch(tmp_path: Path) -> None:
    config = load()

    sft_config = build_multitask_sft_config(config, tmp_path, seed=7, epoch_steps=1000)

    assert sft_config.num_train_epochs == 1
    assert sft_config.eval_strategy == "steps" and sft_config.eval_steps == 1000
    assert sft_config.save_steps == 1000
    assert sft_config.metric_for_best_model == "eval_combined_loss"
    assert sft_config.load_best_model_at_end is True
    assert sft_config.learning_rate == config.finetune.learning_rate
    assert sft_config.per_device_train_batch_size == config.finetune.batch_size
    assert sft_config.completion_only_loss is True
    assert sft_config.seed == 7


def test_build_multitask_sft_config_uses_trial_hyperparameters(tmp_path: Path) -> None:
    hyperparameters = {"learning_rate": 1e-4, "batch_size": 32, "lr_scheduler_type": "cosine", "warmup_ratio": 0.05}

    sft_config = build_multitask_sft_config(
        load(), tmp_path, seed=1, epoch_steps=250, hyperparameters=hyperparameters, save_models=False
    )

    assert sft_config.learning_rate == 1e-4
    assert sft_config.per_device_train_batch_size == 32
    assert sft_config.lr_scheduler_type == "cosine"
    assert sft_config.save_strategy == "no"
    assert sft_config.load_best_model_at_end is False
    assert sft_config.eval_steps == 250


def test_build_multitask_sft_config_follows_gradient_checkpointing(tmp_path: Path) -> None:
    config = load()

    sft_config = build_multitask_sft_config(config, tmp_path, seed=1, epoch_steps=10)

    assert sft_config.gradient_checkpointing == config.finetune.gradient_checkpointing


def test_epoch_metrics_collects_each_evaluation() -> None:
    log_history = [
        {"loss": 2.5, "step": 500},
        {"eval_combined_loss": 2.2, "eval_medtext_loss": 2.4, "step": 1000},
        {"eval_combined_loss": 2.0, "eval_medtext_loss": 2.3, "step": 2000},
    ]

    curves = epoch_metrics(log_history, ["combined", "medtext"])

    assert curves == {"combined": [2.2, 2.0], "medtext": [2.4, 2.3]}


def test_summarize_seeds_reports_mean_and_std() -> None:
    results = [
        {"test_pubmedqa_accuracy": 0.6, "run_id": "a"},
        {"test_pubmedqa_accuracy": 0.7, "run_id": "b"},
        {"test_pubmedqa_accuracy": 0.8, "run_id": "c"},
    ]

    summary = summarize_seeds(results)

    assert summary["test_pubmedqa_accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["test_pubmedqa_accuracy"]["std"] == pytest.approx(0.1)
    assert "run_id" not in summary


def test_base_model_id_starts_dapt_from_the_pretrained_model() -> None:
    config = load()

    assert base_model_id(config, "original") == config.model.model_id
    assert base_model_id(config, "dapt") == "models:/qwen3-1.7b-pubmed-dapt@candidate"


def test_base_model_id_rejects_an_unknown_base() -> None:
    with pytest.raises(ValueError):
        base_model_id(load(), "instruct")


def test_finetune_stage_names_the_method_and_the_starting_model() -> None:
    assert finetune_stage("lora", "original") == "finetune_lora_original"
