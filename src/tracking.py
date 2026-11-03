"""MLflow tracking for searches, trials, best models and evaluations.

Run layout:
    <kind>-search            parent run: config, datasets, best result (stage=finetune)
      <kind>-trial-N         nested run per trial: hyperparameters and loss curves
    evaluate-<model>-<split> one run per evaluation (stage=evaluate)

Only the best trial of a search stores its model, which is also registered in
the Model Registry under `config.mlflow.registered_model_name`.
"""

import os
import subprocess
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
from datasets import Dataset

from src.config import Config, Hyperparameters, PretrainConfig


def configure_tracking(config: Config | PretrainConfig) -> None:
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
REPO_DIR = Path(__file__).resolve().parent


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
    config: Config | PretrainConfig,
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


def log_trial_parameters(
    search_kind: str, trial_number: int, hyperparameters: Hyperparameters, config: Config
) -> None:
    """Record the chosen and fixed settings before training starts."""
    training = config.training
    mlflow.set_tags({"search_kind": search_kind, "trial_number": str(trial_number)})
    mlflow.log_params(
        {
            "model_id": config.model.model_id,
            "dataset_id": config.data.dataset_id,
            "seed": config.seed,
            "max_seq_len": config.model.max_seq_len,
            "max_epochs": training.max_epochs,
            "early_stopping_patience": training.early_stopping_patience,
            "eval_batch_size": training.eval_batch_size,
            "max_grad_norm": training.max_grad_norm,
            "learning_rate": hyperparameters["learning_rate"],
            "batch_size": hyperparameters["batch_size"],
            "weight_decay": hyperparameters.get("weight_decay", training.weight_decay),
            "lr_scheduler_type": hyperparameters.get(
                "lr_scheduler_type", training.lr_scheduler_type
            ),
            "warmup_ratio": hyperparameters.get("warmup_ratio", training.warmup_ratio),
        }
    )


def log_trial_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Config,
    splits: tuple[str, ...] = ("train", "validation"),
) -> None:
    """Attach the actual prompt-completion splits as MLflow dataset inputs."""
    dataset_name = Path(config.data.processed_dir).name
    for split, dataset, context in (
        ("train", train_dataset, "training"),
        ("validation", val_dataset, "validation"),
    ):
        if split not in splits:
            continue
        dataset_input = mlflow.data.from_huggingface(
            dataset,
            path=config.data.dataset_id,
            name=f"{dataset_name}_{split}_sft",
            digest=dataset._fingerprint,
        )
        mlflow.log_input(
            dataset_input,
            context=context,
            tags={"processed_dir": config.data.processed_dir},
        )


def log_search_inputs(train_dataset: Dataset, val_dataset: Dataset, config: Config) -> None:
    """Record the full configuration and both datasets once, on the search run."""
    mlflow.log_dict(asdict(config), "config.json")
    log_trial_datasets(train_dataset, val_dataset, config)


def log_trial_metrics(
    log_history: list[dict[str, Any]], best_val_loss: float, best_epoch: int
) -> None:
    """Record the curves and the selection metric for this trial."""
    for log in log_history:
        step = int(log.get("epoch", 0))
        if "loss" in log:
            mlflow.log_metric("train_loss", log["loss"], step=step)
        if "eval_loss" in log:
            mlflow.log_metric("val_loss", log["eval_loss"], step=step)
    mlflow.log_metrics({"best_val_loss": best_val_loss, "best_epoch": best_epoch})


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
    model_source: str, split: str, metrics: dict[str, float], config: Config
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
