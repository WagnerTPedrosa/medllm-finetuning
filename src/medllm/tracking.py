"""MLflow tracking: experiments, stage runs (tagged with the git commit), and the Model Registry.

Every pipeline stage opens a run with `start_stage_run`; the best model of a
stage is logged and registered under `config.mlflow.registered_model_name`
with an alias (`log_best_model`).
"""

import os
import subprocess
import warnings
from importlib.metadata import version
from pathlib import Path

import mlflow

from medllm.config import MultitaskConfig, PretrainConfig


def configure_tracking(config: MultitaskConfig | PretrainConfig) -> None:
    """Use the configured store and avoid names reserved by deleted experiments."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri))
    if config.mlflow.log_system_metrics:
        mlflow.enable_system_metrics_logging()

    requested_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", config.mlflow.experiment_name
    )
    experiment = mlflow.get_experiment_by_name(requested_name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        base_name = f"{requested_name}-rerun"
        experiment_name = base_name
        suffix = 2
        while (
            candidate := mlflow.get_experiment_by_name(experiment_name)
        ) is not None and candidate.lifecycle_stage == "deleted":
            experiment_name = f"{base_name}-{suffix}"
            suffix += 1
        warnings.warn(
            f"MLflow experiment {requested_name!r} was deleted; "
            f"using {experiment_name!r} for new runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        experiment_name = requested_name
    mlflow.set_experiment(experiment_name)


# MLflow reserves these alias names (case insensitive)
RESERVED_ALIASES = {"latest"}


# The repository that contains this code, wherever the process was started from
REPO_DIR = Path(__file__).resolve().parents[2]


def git_tags(repo_dir: str | Path = REPO_DIR) -> dict[str, str]:
    """Commit hash and whether the working tree has uncommitted changes.

    MLflow does not record the commit for notebook runs, so it is tagged here.

    Args:
        repo_dir: Any directory inside the repository (defaults to this code's).

    Returns:
        `git_commit` and `git_dirty` ("true"/"false"), or an empty dict when
        git is unavailable or the directory is not in a repository.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    return {"git_commit": commit, "git_dirty": str(bool(status.strip())).lower()}


def start_stage_run(
    stage: str,
    run_name: str,
    config: MultitaskConfig | PretrainConfig,
    extra_tags: dict[str, str] | None = None,
    nested: bool = False,
) -> mlflow.ActiveRun:
    """Start a run tagged with its pipeline stage, base model and git commit.

    Args:
        stage: Pipeline stage (`pretrain`, `finetune` or `evaluate`).
        run_name: Display name of the run.
        config: Experiment configuration (provides the base model).
        extra_tags: Additional tags.
        nested: Start the run inside the currently active run.

    Returns:
        The active run (use it as a context manager).
    """
    tags: dict[str, str] = {
        "stage": stage,
        "base_model": config.model.model_id,
        **git_tags(),
        **(extra_tags or {}),
    }
    return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)


def transformers_pip_requirements() -> list[str]:
    """Pinned packages needed to load a logged transformers model."""
    return [
        f"transformers=={version('transformers')}",
        f"torch=={version('torch')}",
        f"safetensors=={version('safetensors')}",
    ]


def log_saved_model(model_dir: Path) -> str:
    """Register an existing Hugging Face checkpoint as a logged model."""
    model_info = mlflow.transformers.log_model(
        transformers_model=str(model_dir),
        task="text-generation",
        name="model",
        pip_requirements=transformers_pip_requirements(),
    )
    mlflow.set_tag("mlflow_model_id", model_info.model_id)
    return model_info.model_id


def log_best_model(
    run_id: str, model_dir: str | Path, registered_model_name: str, alias: str
) -> str:
    """Store the best trial's checkpoint in its run and register a new version.

    The new version gets `alias` (for example `grid-best`), so later stages
    can load it as `models:/<registered_model_name>@<alias>`. Promoting a
    version to `champion` is left as a deliberate, manual step.

    Args:
        run_id: Run of the best trial (finished runs are reopened).
        model_dir: Saved Hugging Face checkpoint of that trial.
        registered_model_name: Model Registry name.
        alias: Alias pointing to the new version.

    Returns:
        The MLflow logged-model id.

    Raises:
        ValueError: If the alias is reserved by MLflow (checked before the
            model is logged, so no time is spent on a model that cannot be tagged).
    """
    if alias.lower() in RESERVED_ALIASES:
        raise ValueError(f"MLflow alias {alias!r} is reserved; choose another name")

    parent = mlflow.active_run()
    with mlflow.start_run(run_id=run_id, nested=parent is not None):
        model_info = mlflow.transformers.log_model(
            transformers_model=str(model_dir),
            task="text-generation",
            name="model",
            pip_requirements=transformers_pip_requirements(),
            registered_model_name=registered_model_name,
        )
        mlflow.set_tag("mlflow_model_id", model_info.model_id)

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        registered_model_name, alias, model_info.registered_model_version
    )
    if parent is not None:
        client.set_tag(parent.info.run_id, "best_trial_run_id", run_id)
        client.set_tag(parent.info.run_id, "best_model_id", model_info.model_id)
    return model_info.model_id


def log_evaluation(
    model_source: str, split: str, metrics: dict[str, float], config: MultitaskConfig | PretrainConfig
) -> str:
    """Record an evaluation in its own run (stage=evaluate).

    Args:
        model_source: Evaluated model (Hub id, local path or `models:/` URI).
        split: Evaluated split, used as metric prefix (`test_loss`, ...).
        metrics: Metric values without prefix, e.g. `{"loss": 2.07}`.
        config: Experiment configuration.

    Returns:
        The evaluation run id.
    """
    run_name = f"evaluate-{Path(model_source).name}-{split}"
    tags = {"evaluated_model": model_source, "split": split}
    with start_stage_run(
        "evaluate", run_name, config, tags, nested=mlflow.active_run() is not None
    ) as run:
        mlflow.log_metrics({f"{split}_{name}": value for name, value in metrics.items()})
    return run.info.run_id
