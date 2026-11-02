from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.transformers as mlflow_transformers
import pytest
from datasets import Dataset

from src.config import Config
from src.modeling import resolve_model_source
from src.tracking import (
    configure_tracking,
    git_tags,
    log_best_model,
    log_evaluation,
    log_search_inputs,
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
    config: Config, tracking_store: Path
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


def test_log_search_inputs_records_config_and_datasets(
    config: Config, tracking_store: Path
) -> None:
    configure_tracking(config)
    dataset = Dataset.from_dict({"prompt": ["Question: a"], "completion": ["b"]})

    with start_stage_run("finetune", "grid-search", config) as parent:
        log_search_inputs(dataset, dataset, config)

    saved = mlflow.get_run(parent.info.run_id)
    artifacts = {item.path for item in mlflow.MlflowClient().list_artifacts(parent.info.run_id)}
    assert "config.json" in artifacts
    assert len(saved.inputs.dataset_inputs) == 2


def test_log_best_model_registers_a_version_with_alias(
    config: Config, tracking_store: Path, monkeypatch: pytest.MonkeyPatch
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


def test_log_evaluation_records_prefixed_metrics(config: Config, tracking_store: Path) -> None:
    configure_tracking(config)

    run_id = log_evaluation(
        "checkpoints/gpt2-medtext-best", "test", {"loss": 2.0, "perplexity": 7.39}, config
    )

    saved = mlflow.get_run(run_id)
    assert saved.data.metrics == {"test_loss": 2.0, "test_perplexity": 7.39}
    assert saved.data.tags["stage"] == "evaluate"
    assert saved.data.tags["evaluated_model"] == "checkpoints/gpt2-medtext-best"


def test_resolve_model_source_keeps_hub_ids_and_local_paths() -> None:
    assert resolve_model_source("openai-community/gpt2") == "openai-community/gpt2"
    assert resolve_model_source("checkpoints/gpt2-medtext-best") == "checkpoints/gpt2-medtext-best"


def test_resolve_model_source_downloads_logged_model_weights(
    config: Config, tracking_store: Path
) -> None:
    configure_tracking(config)
    packaged = tracking_store / "packaged"
    (packaged / "model").mkdir(parents=True)
    (packaged / "model" / "config.json").write_text('{"model_type": "gpt2"}')
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(packaged), "packaged")

    local_dir = Path(resolve_model_source(f"runs:/{run.info.run_id}/packaged"))

    assert local_dir.name == "model"
    assert (local_dir / "config.json").exists()
