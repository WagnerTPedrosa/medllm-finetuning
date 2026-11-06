from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.transformers as mlflow_transformers
import pytest

from medllm.config import MultitaskConfig
from medllm.tracking import (
    SectionedMLflowCallback,
    configure_tracking,
    git_tags,
    log_best_model,
    log_evaluation,
    metric_section_name,
    sectioned_metrics,
    start_stage_run,
)

REPO_DIR = Path(__file__).resolve().parents[1]


def test_git_tags_reports_commit_and_dirty_flag() -> None:
    tags = git_tags(REPO_DIR)

    assert len(tags["git_commit"]) == 40
    assert tags["git_dirty"] in {"true", "false"}


def test_git_tags_is_empty_outside_a_repository(tmp_path: Path) -> None:
    assert git_tags(tmp_path) == {}


def test_trial_runs_are_nested_under_the_search_run(
    config: MultitaskConfig, tracking_store: Path
) -> None:
    configure_tracking(config)

    with start_stage_run("finetune", "grid-search", config, {"search_kind": "grid"}) as parent:
        with start_stage_run("finetune", "grid-trial-1", config, nested=True) as child:
            pass

    child_tags = mlflow.get_run(child.info.run_id).data.tags
    parent_tags = mlflow.get_run(parent.info.run_id).data.tags
    assert child_tags["mlflow.parentRunId"] == parent.info.run_id
    assert parent_tags["stage"] == "finetune"
    assert parent_tags["search_kind"] == "grid"
    assert parent_tags["base_model"] == config.model.model_id
    assert "git_commit" in parent_tags


def test_log_best_model_registers_a_version_with_alias(
    config: MultitaskConfig, tracking_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_tracking(config)
    client = mlflow.MlflowClient()
    calls: dict[str, str] = {}

    def fake_log_model(**kwargs: object) -> SimpleNamespace:
        name = str(kwargs["registered_model_name"])
        calls["run_id"] = mlflow.active_run().info.run_id
        calls["source"] = str(kwargs["transformers_model"])
        if not client.search_registered_models(filter_string=f"name = '{name}'"):
            client.create_registered_model(name)
        version = client.create_model_version(name, source=calls["source"], run_id=calls["run_id"])
        return SimpleNamespace(model_id="m-test", registered_model_version=version.version)

    monkeypatch.setattr(mlflow_transformers, "log_model", fake_log_model)
    model_dir = tracking_store / "best"
    model_dir.mkdir()

    with start_stage_run("finetune", "grid-search", config) as parent:
        with start_stage_run("finetune", "grid-trial-2", config, nested=True) as best_trial:
            pass
        model_id = log_best_model(best_trial.info.run_id, model_dir, "test-sft", "grid-best")

    assert model_id == "m-test"
    assert calls["run_id"] == best_trial.info.run_id
    assert int(client.get_model_version_by_alias("test-sft", "grid-best").version) == 1
    assert mlflow.get_run(parent.info.run_id).data.tags["best_trial_run_id"] == (
        best_trial.info.run_id
    )


def test_log_evaluation_records_prefixed_metrics(config: MultitaskConfig, tracking_store: Path) -> None:
    configure_tracking(config)

    run_id = log_evaluation(
        "checkpoints/gpt2-medtext-best", "test", {"loss": 2.0, "perplexity": 7.39}, config
    )

    saved = mlflow.get_run(run_id)
    assert saved.data.metrics == {"test/loss": 2.0, "test/perplexity": 7.39}
    assert saved.data.tags["stage"] == "evaluate"
    assert saved.data.tags["evaluated_model"] == "checkpoints/gpt2-medtext-best"


@pytest.mark.parametrize("alias", ["latest", "Latest"])
def test_log_best_model_rejects_reserved_alias_before_logging(
    alias: str, config: MultitaskConfig, tracking_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(mlflow_transformers, "log_model", lambda **kwargs: calls.append(kwargs))
    configure_tracking(config)
    with mlflow.start_run() as run:
        pass

    with pytest.raises(ValueError, match="reserved"):
        log_best_model(run.info.run_id, tracking_store, "test-sft", alias)

    assert calls == []


def test_configure_tracking_uses_new_name_after_experiment_deleted(
    config: MultitaskConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # Training logs of the Trainer
        ("loss", "train/loss"),
        ("grad_norm", "train/grad_norm"),
        ("learning_rate", "train/learning_rate"),
        ("mean_token_accuracy", "train/mean_token_accuracy"),
        ("epoch", "train/epoch"),
        # Validation during training
        ("eval_loss", "eval/loss"),
        ("eval_combined_loss", "eval/combined_loss"),
        ("eval_perplexity", "eval/perplexity"),
        ("final_eval_loss", "eval/final_loss"),
        # Speed, not quality
        ("eval_combined_runtime", "perf/eval_combined_runtime"),
        ("eval_medqa_samples_per_second", "perf/eval_medqa_samples_per_second"),
        ("train_steps_per_second", "perf/train_steps_per_second"),
        ("total_flos", "perf/total_flos"),
        # Final scores
        ("test_medqa_accuracy", "test/medqa_accuracy"),
        ("test_medqa_accuracy_mean", "test/medqa_accuracy_mean"),
        ("validation_combined_loss", "validation/combined_loss"),
        ("benchmark_medqa_4options_acc", "benchmark/medqa_4options_acc"),
        ("test_medtext_geval_correctness", "geval/medtext_correctness"),
        ("test_medtext_geval_correctness_mean", "geval/medtext_correctness_mean"),
        # One value per run
        ("train_loss", "summary/train_loss"),
        ("best_validation_combined_loss", "summary/best_validation_combined_loss"),
        ("mixture_epochs_trained", "summary/mixture_epochs_trained"),
        ("mixture_epochs_trained_mean", "summary/mixture_epochs_trained_mean"),
        # Already in a section, or unknown
        ("system/gpu_0_utilization_percentage", "system/gpu_0_utilization_percentage"),
        ("custom_metric", "custom_metric"),
    ],
)
def test_metric_section_name_groups_metrics_by_prefix(key: str, expected: str) -> None:
    assert metric_section_name(key) == expected


def test_sectioned_metrics_renames_every_key_and_keeps_values() -> None:
    assert sectioned_metrics({"loss": 1.5, "test_medqa_accuracy": 0.4}) == {
        "train/loss": 1.5,
        "test/medqa_accuracy": 0.4,
    }


class RecordingMlflow:
    """Stands in for the `mlflow` module inside the Trainer callback."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((metrics, step))


def test_sectioned_mlflow_callback_logs_trainer_metrics_by_section() -> None:
    recorder = RecordingMlflow()
    callback = SectionedMLflowCallback()
    # Skip the callback's setup, which reads the TrainingArguments and the model
    callback._initialized = True
    callback._async_log = False
    callback._ml_flow = recorder
    state = SimpleNamespace(is_world_process_zero=True, global_step=500)

    callback.on_log(None, state, None, logs={"loss": 1.9, "eval_combined_loss": 1.5})

    assert recorder.logged == [({"train/loss": 1.9, "eval/combined_loss": 1.5}, 500)]
