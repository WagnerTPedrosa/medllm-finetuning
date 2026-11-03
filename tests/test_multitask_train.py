from pathlib import Path

import optuna
import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from src.config import FloatRange, MultitaskConfig, load_multitask_config
from src.multitask_train import (
    StopAfterStepsCallback,
    base_model_id,
    build_multitask_sft_config,
    epoch_metrics,
    steps_per_epoch,
    summarize_seeds,
    sweep_eval_steps,
    sweep_results,
)
from src.train import OptunaPruningCallback



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


class FakeTrial:
    """Records reports; prunes when told to."""

    def __init__(self, prune_after: int | None = None) -> None:
        self.reports: list[tuple[int, float]] = []
        self.prune_after = prune_after

    def report(self, value: float, step: int) -> None:
        self.reports.append((step, value))

    def should_prune(self) -> bool:
        return self.prune_after is not None and len(self.reports) >= self.prune_after


def evaluate_callback(callback: OptunaPruningCallback, metrics: dict[str, float]) -> None:
    callback.on_evaluate(
        TrainingArguments(output_dir="unused", report_to="none"),
        TrainerState(),
        TrainerControl(),
        metrics=metrics,
    )


def test_load_multitask_config_reads_sweep_section() -> None:
    sweep = load().sweep

    assert sweep.n_trials == 12
    assert isinstance(sweep.search_space["learning_rate"], FloatRange)
    assert sweep.search_space["batch_size"] == [8, 16, 32]


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


def test_sweep_eval_steps_spreads_evaluations_over_the_budget() -> None:
    # 1,000 examples, batch 10 -> 100 steps; 4 evaluations -> every 25 steps
    assert sweep_eval_steps(num_examples=1000, batch_size=10, evaluations=4) == 25


def test_pruning_callback_only_reports_the_configured_metric() -> None:
    trial = FakeTrial()
    callback = OptunaPruningCallback(trial, metric_name="eval_combined_loss")

    evaluate_callback(callback, {"eval_medtext_loss": 2.5})
    evaluate_callback(callback, {"eval_combined_loss": 2.2})
    evaluate_callback(callback, {"eval_combined_loss": 2.0})

    # Steps start at 0, so MedianPruner(n_warmup_steps=1) never prunes at the first evaluation
    assert trial.reports == [(0, 2.2), (1, 2.0)]


def test_pruning_callback_raises_when_the_trial_should_stop() -> None:
    callback = OptunaPruningCallback(FakeTrial(prune_after=1), metric_name="eval_combined_loss")

    with pytest.raises(optuna.TrialPruned):
        evaluate_callback(callback, {"eval_combined_loss": 3.0})


def test_sweep_results_lists_complete_and_pruned_trials() -> None:
    study = optuna.create_study(direction="minimize")
    distributions = {"batch_size": optuna.distributions.CategoricalDistribution([8, 16])}
    study.add_trial(
        optuna.trial.create_trial(
            params={"batch_size": 16},
            distributions=distributions,
            value=2.0,
            user_attrs={"dev_combined_curve": [2.3, 2.0], "mlflow_run_id": "a"},
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"batch_size": 8},
            distributions=distributions,
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={"dev_combined_curve": [2.9], "mlflow_run_id": "b"},
        )
    )

    rows = sweep_results(study)

    assert [row["status"] for row in rows] == ["complete", "pruned"]
    assert rows[0]["best_dev_combined_loss"] == 2.0
    assert rows[1]["best_dev_combined_loss"] == 2.9
    assert rows[0]["batch_size"] == 16


def test_stop_after_steps_callback_ends_training_at_the_budget() -> None:
    callback = StopAfterStepsCallback(max_steps=100)
    state, control = TrainerState(), TrainerControl()
    args = TrainingArguments(output_dir="unused", report_to="none")

    state.global_step = 99
    callback.on_step_end(args, state, control)
    assert control.should_training_stop is False

    state.global_step = 100
    callback.on_step_end(args, state, control)
    assert control.should_training_stop is True


def test_base_model_id_starts_dapt_from_the_pretrained_model() -> None:
    config = load()

    assert base_model_id(config, "original") == config.model.model_id
    assert base_model_id(config, "dapt") == "models:/gpt2-pubmed-dapt@candidate"


def test_base_model_id_rejects_an_unknown_base() -> None:
    with pytest.raises(ValueError):
        base_model_id(load(), "instruct")


def test_build_multitask_sft_config_follows_gradient_checkpointing(tmp_path: Path) -> None:
    config = load()

    sft_config = build_multitask_sft_config(config, tmp_path, seed=1, epoch_steps=10)

    assert sft_config.gradient_checkpointing == config.finetune.gradient_checkpointing
