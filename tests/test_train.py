from pathlib import Path

import mlflow
import optuna
import pytest
from datasets import Dataset

from src.config import Config, FloatRange, load_config
from src.train import (
    build_sft_config,
    find_best_result,
    losses_by_epoch,
    study_results,
    suggest_hyperparameters,
    summarize_losses,
)
from src.tracking import (
    configure_tracking,
    log_trial_datasets,
    log_trial_metrics,
    log_trial_parameters,
)

LOG_HISTORY = [
    {"loss": 2.5, "epoch": 1.0},
    {"eval_loss": 2.2, "epoch": 1.0},
    {"loss": 2.1, "epoch": 2.0},
    {"eval_loss": 2.0, "epoch": 2.0},
    {"loss": 1.8, "epoch": 3.0},
    {"eval_loss": 2.1, "epoch": 3.0},
    {"train_runtime": 10.0, "train_loss": 2.1},
]


@pytest.fixture
def config() -> Config:
    return load_config(Path("configs/gpt2_medtext.yaml"))


def test_losses_by_epoch_splits_train_and_validation() -> None:
    train_losses, val_losses = losses_by_epoch(LOG_HISTORY)

    assert train_losses == [2.5, 2.1, 1.8]
    assert val_losses == [2.2, 2.0, 2.1]


def test_summarize_losses_finds_best_epoch() -> None:
    metrics = summarize_losses(LOG_HISTORY)

    assert metrics["best_val_loss"] == 2.0
    assert metrics["best_epoch"] == 2


def test_summarize_losses_raises_without_evaluations() -> None:
    with pytest.raises(ValueError):
        summarize_losses([{"loss": 2.5}])


def test_build_sft_config_uses_config_defaults(config: Config, tmp_path: Path) -> None:
    sft_config = build_sft_config(
        {"learning_rate": 5e-5, "batch_size": 8}, config, tmp_path
    )

    assert sft_config.learning_rate == 5e-5
    assert sft_config.per_device_train_batch_size == 8
    assert sft_config.per_device_eval_batch_size == config.training.eval_batch_size
    assert sft_config.weight_decay == config.training.weight_decay
    assert sft_config.completion_only_loss is True


def test_build_sft_config_lets_hyperparameters_override_defaults(
    config: Config, tmp_path: Path
) -> None:
    sft_config = build_sft_config(
        {"learning_rate": 5e-5, "batch_size": 8, "weight_decay": 0.1}, config, tmp_path
    )

    assert sft_config.weight_decay == 0.1


def test_suggest_hyperparameters_follows_search_space() -> None:
    search_space = {
        "learning_rate": FloatRange(low=1e-5, high=1e-4, log=True),
        "batch_size": [2, 8],
    }
    trial = optuna.trial.FixedTrial({"learning_rate": 3e-5, "batch_size": 8})

    hyperparameters = suggest_hyperparameters(trial, search_space)

    assert hyperparameters == {"learning_rate": 3e-5, "batch_size": 8}


def test_study_results_includes_completed_and_pruned_trials() -> None:
    study = optuna.create_study(direction="minimize")
    distributions = {"batch_size": optuna.distributions.CategoricalDistribution([2, 8])}
    metrics = summarize_losses(LOG_HISTORY)
    study.add_trial(
        optuna.trial.create_trial(
            params={"batch_size": 8},
            distributions=distributions,
            value=2.0,
            user_attrs={"metrics": metrics},
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"batch_size": 2},
            distributions=distributions,
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={"metrics": metrics},
        )
    )

    results = study_results(study)

    assert [result["trial"] for result in results] == [1, 2]
    assert [result["status"] for result in results] == ["complete", "pruned"]
    assert results[0]["batch_size"] == 8
    assert results[0]["best_val_loss"] == 2.0


def test_find_best_result_ignores_pruned_trials() -> None:
    results = [
        {"trial": 1, "status": "complete", "best_val_loss": 2.1},
        {"trial": 2, "status": "pruned", "best_val_loss": 1.5},
        {"trial": 3, "status": "complete", "best_val_loss": 2.0},
    ]

    assert find_best_result(results)["trial"] == 3


def test_find_best_result_raises_without_completed_trials() -> None:
    with pytest.raises(ValueError):
        find_best_result([{"trial": 1, "status": "pruned", "best_val_loss": 2.0}])


def test_mlflow_trial_run_keeps_parameters_metrics_and_datasets(
    config: Config, tracking_store: Path
) -> None:
    configure_tracking(config)
    with mlflow.start_run(run_name="grid-trial-1") as run:
        log_trial_parameters("grid", 1, {"learning_rate": 5e-5, "batch_size": 8}, config)
        dataset = Dataset.from_dict({"prompt": ["Question: a"], "completion": ["b"]})
        log_trial_datasets(dataset, dataset, config)
        log_trial_metrics(LOG_HISTORY, best_val_loss=2.0, best_epoch=2)

    saved = mlflow.get_run(run.info.run_id)
    assert saved.data.params["learning_rate"] == "5e-05"
    assert saved.data.params["weight_decay"] == "0.01"
    assert saved.data.tags["search_kind"] == "grid"
    assert saved.data.metrics["best_val_loss"] == 2.0
    assert len(saved.inputs.dataset_inputs) == 2
    assert [
        metric.value
        for metric in mlflow.MlflowClient().get_metric_history(run.info.run_id, "val_loss")
    ] == [2.2, 2.0, 2.1]
    # Trial runs no longer store a model; only the best trial does (see test_tracking)
    assert mlflow.MlflowClient().list_artifacts(run.info.run_id) == []


def test_configure_tracking_uses_new_name_after_experiment_deleted(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "deleted-test")

    try:
        configure_tracking(config)
        old_experiment = mlflow.get_experiment_by_name("deleted-test")
        mlflow.MlflowClient().delete_experiment(old_experiment.experiment_id)

        with pytest.warns(RuntimeWarning, match="using 'deleted-test-rerun'"):
            configure_tracking(config)
        with mlflow.start_run() as run:
            selected = mlflow.get_experiment(run.info.experiment_id)
            assert selected.name == "deleted-test-rerun"
        assert mlflow.get_experiment(old_experiment.experiment_id).lifecycle_stage == "deleted"
    finally:
        mlflow.set_tracking_uri(old_uri)
