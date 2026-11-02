"""Typed access to the experiment configuration in `configs/*.yaml`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HyperparameterValue = float | int | str
Hyperparameters = dict[str, HyperparameterValue]


@dataclass(frozen=True)
class FloatRange:
    """Continuous search range for one hyperparameter."""

    low: float
    high: float
    log: bool = False


SearchSpace = dict[str, FloatRange | list[HyperparameterValue]]


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    max_seq_len: int


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str
    processed_dir: str
    holdout_fraction: float


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    early_stopping_patience: int
    eval_batch_size: int
    max_grad_norm: float
    weight_decay: float
    lr_scheduler_type: str
    warmup_ratio: float


@dataclass(frozen=True)
class GridSearchConfig:
    param_grid: dict[str, list[HyperparameterValue]]
    best_checkpoint_dir: str
    results_path: str
    trials_dir: str


@dataclass(frozen=True)
class OptunaConfig:
    n_trials: int
    search_space: SearchSpace
    pruner_startup_trials: int
    pruner_warmup_epochs: int
    best_checkpoint_dir: str
    results_path: str
    trials_dir: str


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    log_system_metrics: bool


@dataclass(frozen=True)
class Config:
    seed: int
    mlflow: MLflowConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    grid_search: GridSearchConfig
    optuna: OptunaConfig
    generation: GenerationConfig


def parse_search_space(raw_space: dict[str, Any]) -> SearchSpace:
    """Convert the YAML search space into typed ranges and choices.

    Args:
        raw_space: Mapping from hyperparameter name to either a
            `{low, high, log}` mapping (float range) or a list (categorical).

    Returns:
        The search space with ranges as `FloatRange` and choices as lists.

    Raises:
        ValueError: If a range bound is not a number or an entry has another type.
    """
    search_space: SearchSpace = {}

    for name, space in raw_space.items():
        if isinstance(space, list):
            search_space[name] = space
            continue

        if not isinstance(space, dict):
            raise ValueError(f"{name}: expected a list or a {{low, high}} mapping")

        low, high = space["low"], space["high"]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(
                f"{name}: low and high must be numbers (write 1.0e-5, not 1e-5)"
            )
        search_space[name] = FloatRange(
            low=float(low), high=float(high), log=space.get("log", False)
        )

    return search_space


def load_config(path: str | Path) -> Config:
    """Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The typed configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required section is missing.
        TypeError: If a section has missing or unknown fields.
    """
    with open(path) as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    optuna_section: dict[str, Any] = dict(raw["optuna"])
    optuna_section["search_space"] = parse_search_space(optuna_section["search_space"])

    return Config(
        seed=raw["seed"],
        mlflow=MLflowConfig(**raw["mlflow"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        grid_search=GridSearchConfig(**raw["grid_search"]),
        optuna=OptunaConfig(**optuna_section),
        generation=GenerationConfig(**raw["generation"]),
    )
