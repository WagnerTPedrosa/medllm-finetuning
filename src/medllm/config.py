"""Typed access to the experiment configuration in `configs/`.

The pipeline configuration (`configs/config.yaml` plus one file per base model
in `configs/model/`) is composed by Hydra and loaded into frozen dataclasses:
`load_pretrain_config` for PubMed pretraining, `load_multitask_config` for the
fine-tuning data, training and evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# The repository's config directory, wherever the process was started from
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


FINETUNE_METHODS = ("lora", "full")


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
class TrainingConfig:
    max_epochs: int
    early_stopping_patience: int
    eval_batch_size: int
    max_grad_norm: float
    weight_decay: float
    lr_scheduler_type: str
    warmup_ratio: float


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    log_system_metrics: bool


@dataclass(frozen=True)
class CorpusConfig:
    raw_dir: str
    clean_dir: str
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
    gradient_checkpointing: bool


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
    results_dir: str


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


def compose_config(overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose `configs/config.yaml` with Hydra and resolve every interpolation.

    Args:
        overrides: Hydra overrides, for example `["model=qwen3-1.7b"]`.

    Returns:
        The resolved configuration as plain dicts and lists.

    Raises:
        hydra.errors.MissingConfigException: If an override names an unknown model file.
        omegaconf.errors.MissingMandatoryValue: If a `???` value is left unset.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="config", overrides=overrides or [])
    return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


def load_pretrain_config(overrides: list[str] | None = None) -> PretrainConfig:
    """Continued-pretraining configuration of the selected base model.

    Args:
        overrides: Hydra overrides (see `compose_config`); the default model is GPT-2.

    Returns:
        The typed configuration (with `~` in `corpus.raw_dir` expanded).
    """
    raw: dict[str, Any] = compose_config(overrides)

    corpus_section: dict[str, Any] = dict(raw["corpus"])
    corpus_section["raw_dir"] = str(Path(corpus_section["raw_dir"]).expanduser())

    return PretrainConfig(
        seed=raw["seed"],
        mlflow=MLflowConfig(
            tracking_uri=raw["mlflow"]["tracking_uri"],
            experiment_name=raw["mlflow"]["pretrain_experiment"],
            registered_model_name=raw["mlflow"]["pretrain_registered_model"],
            log_system_metrics=raw["mlflow"]["log_system_metrics"],
        ),
        model=ModelConfig(model_id=raw["model"]["model_id"], max_seq_len=raw["model"]["max_seq_len"]),
        corpus=CorpusConfig(**corpus_section),
        pretraining=PretrainingConfig(**raw["pretraining"]),
        lr_sweep=LRSweepConfig(**raw["lr_sweep"]),
        results_dir=raw["results_dir"],
    )


@dataclass(frozen=True)
class SourceDataset:
    """A Hugging Face dataset (and subset) at a fixed revision."""

    dataset_id: str
    subset: str | None
    revision: str


@dataclass(frozen=True)
class DatasetsConfig:
    raw_dir: str
    sources: dict[str, SourceDataset]


@dataclass(frozen=True)
class MultitaskDataConfig:
    preprocessed_dir: str
    splits_dir: str
    processed_dir: str
    pubmedqa_test_ground_truth: str
    medtext_holdout_fraction: float
    pubmedqa_validation_size: int
    medmcqa_validation_size: int
    medqa_validation_size: int
    num_proc: int
    system_prompt: str


@dataclass(frozen=True)
class MixtureConfig:
    """How the unique training examples are repeated and sampled in each epoch."""

    epochs: int
    medtext_repeats: int
    pubmedqa_labeled_repeats: int
    pubmedqa_artificial_per_class: int
    medmcqa_per_epoch: int
    medqa_repeats: int


@dataclass(frozen=True)
class DataFilterConfig:
    """Tokenizer and length limit that decide which fine-tuning examples are dropped."""

    tokenizer: str
    max_seq_len: int


@dataclass(frozen=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    modules_to_save: list[str]


@dataclass(frozen=True)
class FinetuneConfig:
    method: str
    learning_rate: float
    batch_size: int
    gradient_checkpointing: bool
    seeds: list[int]
    lora: LoraSettings
    output_dir: str
    dapt_model_id: str


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
    vllm_bin: str
    serve_args: list[str]
    startup_timeout: int
    server_log: str
    max_new_tokens: int
    max_concurrent: int
    num_examples: int | None


@dataclass(frozen=True)
class BenchmarksConfig:
    """lm-evaluation-harness tasks and how they are run."""

    tasks: list[str]
    num_fewshot: int
    batch_size: int


@dataclass(frozen=True)
class MultitaskConfig:
    seed: int
    mlflow: MLflowConfig
    model: ModelConfig
    training: TrainingConfig
    data_filter: DataFilterConfig
    datasets: DatasetsConfig
    multitask: MultitaskDataConfig
    mixture: MixtureConfig
    finetune: FinetuneConfig
    sweep: MultitaskSweepConfig
    benchmarks: BenchmarksConfig
    geval: GEvalConfig
    results_dir: str


def load_multitask_config(overrides: list[str] | None = None) -> MultitaskConfig:
    """Multitask fine-tuning and G-Eval configuration of the selected base model.

    Args:
        overrides: Hydra overrides (see `compose_config`); the default model is GPT-2.

    Returns:
        The typed configuration (with `~` in `geval.vllm_bin` expanded).

    Raises:
        ValueError: If `finetune.method` is not one of `FINETUNE_METHODS`.
    """
    raw: dict[str, Any] = compose_config(overrides)

    sweep_section: dict[str, Any] = dict(raw["sweep"])
    sweep_section["search_space"] = parse_search_space(sweep_section["search_space"])
    finetune_section: dict[str, Any] = dict(raw["finetune"])
    finetune_section["lora"] = LoraSettings(**finetune_section["lora"])
    if finetune_section["method"] not in FINETUNE_METHODS:
        raise ValueError(f"finetune.method must be one of {FINETUNE_METHODS}, got {finetune_section['method']!r}")
    geval_section: dict[str, Any] = dict(raw["geval"])
    geval_section["vllm_bin"] = str(Path(geval_section["vllm_bin"]).expanduser())

    return MultitaskConfig(
        seed=raw["seed"],
        mlflow=MLflowConfig(
            tracking_uri=raw["mlflow"]["tracking_uri"],
            experiment_name=raw["mlflow"]["multitask_experiment"],
            registered_model_name=raw["mlflow"]["multitask_registered_model"],
            log_system_metrics=raw["mlflow"]["log_system_metrics"],
        ),
        model=ModelConfig(model_id=raw["model"]["model_id"], max_seq_len=raw["model"]["max_seq_len"]),
        training=TrainingConfig(**raw["training"]),
        data_filter=DataFilterConfig(**raw["data_filter"]),
        datasets=DatasetsConfig(
            raw_dir=raw["datasets"]["raw_dir"],
            sources={name: SourceDataset(**source) for name, source in raw["datasets"]["sources"].items()},
        ),
        multitask=MultitaskDataConfig(**raw["multitask"]),
        mixture=MixtureConfig(**raw["mixture"]),
        finetune=FinetuneConfig(**finetune_section),
        sweep=MultitaskSweepConfig(**sweep_section),
        benchmarks=BenchmarksConfig(**raw["benchmarks"]),
        geval=GEvalConfig(**geval_section),
        results_dir=raw["results_dir"],
    )
