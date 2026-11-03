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


@dataclass(frozen=True)
class CorpusConfig:
    raw_dir: str
    processed_dir: str
    validation_fraction: float
    remove_boilerplate_sections: bool
    remove_citations: bool
    benchmark_dataset_id: str
    benchmark_subset: str
    num_proc: int


@dataclass(frozen=True)
class PretrainingConfig:
    output_dir: str
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    num_train_epochs: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    logging_steps: int
    bf16: bool
    torch_compile: bool
    optim: str
    dataloader_num_workers: int
    registry_alias: str


@dataclass(frozen=True)
class LRSweepConfig:
    learning_rates: list[float]
    max_steps: int


@dataclass(frozen=True)
class PretrainConfig:
    seed: int
    mlflow: MLflowConfig
    model: ModelConfig
    corpus: CorpusConfig
    pretraining: PretrainingConfig
    lr_sweep: LRSweepConfig


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


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    """Load a continued-pretraining configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The typed configuration (with `~` in `corpus.raw_dir` expanded).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required section is missing.
        TypeError: If a section has missing or unknown fields.
    """
    with open(path) as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    corpus_section: dict[str, Any] = dict(raw["corpus"])
    corpus_section["raw_dir"] = str(Path(corpus_section["raw_dir"]).expanduser())

    return PretrainConfig(
        seed=raw["seed"],
        mlflow=MLflowConfig(**raw["mlflow"]),
        model=ModelConfig(**raw["model"]),
        corpus=CorpusConfig(**corpus_section),
        pretraining=PretrainingConfig(**raw["pretraining"]),
        lr_sweep=LRSweepConfig(**raw["lr_sweep"]),
    )


@dataclass(frozen=True)
class MultitaskDataConfig:
    processed_dir: str
    medtext_dir: str
    pubmedqa_dataset_id: str
    pubmedqa_test_ground_truth: str
    pubmedqa_validation_size: int
    medmcqa_dataset_id: str
    medmcqa_dev_fraction: float
    medtext_repeats: int
    pubmedqa_labeled_repeats: int
    pubmedqa_artificial_per_class: int
    medmcqa_per_epoch: int
    dev_medmcqa_size: int
    num_proc: int


@dataclass(frozen=True)
class FinetuneConfig:
    learning_rate: float
    batch_size: int
    seeds: list[int]
    output_dir: str


@dataclass(frozen=True)
class MultitaskSweepConfig:
    n_trials: int
    search_space: SearchSpace
    train_epochs: float
    evals_per_epoch: int
    pruner_startup_trials: int
    pruner_warmup_evals: int
    results_path: str
    trials_dir: str


@dataclass(frozen=True)
class GEvalConfig:
    judge_model: str
    base_url: str
    max_new_tokens: int
    max_concurrent: int
    num_examples: int | None


@dataclass(frozen=True)
class MultitaskConfig:
    seed: int
    mlflow: MLflowConfig
    model: ModelConfig
    training: TrainingConfig
    multitask: MultitaskDataConfig
    finetune: FinetuneConfig
    sweep: MultitaskSweepConfig
    geval: GEvalConfig


def load_multitask_config(path: str | Path) -> MultitaskConfig:
    """Load a multitask fine-tuning configuration from a YAML file.

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

    sweep_section: dict[str, Any] = dict(raw["sweep"])
    sweep_section["search_space"] = parse_search_space(sweep_section["search_space"])

    return MultitaskConfig(
        seed=raw["seed"],
        mlflow=MLflowConfig(**raw["mlflow"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        multitask=MultitaskDataConfig(**raw["multitask"]),
        finetune=FinetuneConfig(**raw["finetune"]),
        sweep=MultitaskSweepConfig(**sweep_section),
        geval=GEvalConfig(**raw["geval"]),
    )
