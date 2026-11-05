
import optuna
import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from medllm.config import FloatRange, MultitaskConfig, load_multitask_config
from medllm.training.sweep import (
    OptunaPruningCallback,
    StopAfterStepsCallback,
    suggest_hyperparameters,
    sweep_eval_steps,
    sweep_results,
)


def load() -> MultitaskConfig:
    return load_multitask_config()


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
            user_attrs={"validation_combined_curve": [2.3, 2.0], "mlflow_run_id": "a"},
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"batch_size": 8},
            distributions=distributions,
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={"validation_combined_curve": [2.9], "mlflow_run_id": "b"},
        )
    )

    rows = sweep_results(study)

    assert [row["status"] for row in rows] == ["complete", "pruned"]
    assert rows[0]["best_validation_combined_loss"] == 2.0
    assert rows[1]["best_validation_combined_loss"] == 2.9
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


def test_suggest_hyperparameters_follows_search_space() -> None:
    search_space = {
        "learning_rate": FloatRange(low=1e-5, high=1e-4, log=True),
        "batch_size": [2, 8],
    }
    trial = optuna.trial.FixedTrial({"learning_rate": 3e-5, "batch_size": 8})

    hyperparameters = suggest_hyperparameters(trial, search_space)

    assert hyperparameters == {"learning_rate": 3e-5, "batch_size": 8}
