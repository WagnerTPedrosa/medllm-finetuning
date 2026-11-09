"""MLflow tracking: experiments, stage runs (tagged with the git commit), and the Model Registry.

Every pipeline stage opens a run with `start_stage_run`; the best model of a
stage is logged and registered under `config.mlflow.registered_model_name`
with an alias (`log_best_model`).

Metrics are logged under sections (`train/loss`, `eval/combined_loss`,
`test/medqa_accuracy`, ...): the MLflow UI groups the charts of a run by the
prefix before the slash (`metric_section_name`).
"""

import os
import subprocess
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.integrations import MLflowCallback

from medllm.config import MultitaskConfig, PretrainConfig
from medllm.reporting.report import BASES


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


# Logs of the Trainer while it trains (the rest of its keys say which split they belong to)
TRAIN_KEYS = {"loss", "grad_norm", "learning_rate", "entropy", "mean_token_accuracy", "num_tokens", "epoch"}
# One value per run, logged at the end of training
SUMMARY_KEYS = {"train_loss", "mixture_epochs_trained"}
# Speed, not model quality
PERF_SUFFIXES = ("_runtime", "_samples_per_second", "_steps_per_second")
# Metric prefix -> MLflow section, checked in order
SECTION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("final_eval_", "eval/final_"),
    ("eval_", "eval/"),
    ("test_", "test/"),
    ("validation_", "validation/"),
    ("benchmark_", "benchmark/"),
    ("best_", "summary/best_"),
)
GEVAL_MARKER = "_geval_"
# Mean and standard deviation over seeds, logged to the parent run
SEED_SUMMARY_SUFFIXES = ("_mean", "_std")


def metric_section_name(key: str) -> str:
    """MLflow name of a metric, with its section as prefix (`eval_combined_loss` -> `eval/combined_loss`).

    Sections: `train` (training logs), `eval` (validation during training),
    `test`, `validation`, `benchmark`, `geval` (scores of the judge model),
    `perf` (runtime and throughput) and `summary` (one value per run).

    Args:
        key: Metric name without a section, as the code computes it.

    Returns:
        The name with its section; names that already have one, or match no
        section, are returned unchanged.
    """
    if "/" in key:
        return key
    if key in TRAIN_KEYS:
        return f"train/{key}"
    # mixture_epochs_trained_mean is a summary like mixture_epochs_trained
    base_key: str = key
    for suffix in SEED_SUMMARY_SUFFIXES:
        base_key = base_key.removesuffix(suffix)
    if base_key in SUMMARY_KEYS:
        return f"summary/{key}"
    if key == "total_flos" or key.endswith(PERF_SUFFIXES):
        return f"perf/{key}"
    if GEVAL_MARKER in key:
        # test_medtext_geval_correctness -> geval/medtext_correctness
        return "geval/" + key.removeprefix("test_").replace(GEVAL_MARKER, "_")
    for prefix, section in SECTION_PREFIXES:
        if key.startswith(prefix):
            return section + key.removeprefix(prefix)
    return key


def sectioned_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """The same metrics, named with their section (see `metric_section_name`)."""
    return {metric_section_name(key): value for key, value in metrics.items()}


class SectionedMLflowCallback(MLflowCallback):
    """The Trainer's MLflow logging, with every metric named by its section.

    Use it with `report_to="none"`, so the default callback does not log the
    same metrics again under their plain names.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().on_log(args, state, control, sectioned_metrics(logs), model=model, **kwargs)


def run_variant(base: str, method: str | None) -> str:
    """Training recipe of a model, for the `variant` tag of its runs.

    Args:
        base: `original` (the Hub model) or `dapt` (after PubMed pretraining).
        method: Fine-tuning method (`lora`, `full`), or None for a model
            that is not fine-tuned.

    Returns:
        `base`, `dapt`, `lora`, `dapt-lora`, `full` or `dapt-full`.

    Raises:
        ValueError: If `base` is not one of `BASES`.
    """
    if base not in BASES:
        raise ValueError(f"base must be one of {BASES}, got {base!r}")
    if method is None:
        return "base" if base == "original" else "dapt"
    return method if base == "original" else f"dapt-{method}"


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
    """Start a run tagged with its pipeline stage, model, base model and git commit.

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
        # Experiments are shared by every model: this tag tells their runs apart
        "model": config.model.name,
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
        split: Evaluated split, used as metric section (`test/loss`, ...).
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
        mlflow.log_metrics(sectioned_metrics({f"{split}_{name}": value for name, value in metrics.items()}))
    return run.info.run_id
