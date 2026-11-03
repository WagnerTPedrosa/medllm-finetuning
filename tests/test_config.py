from pathlib import Path

import pytest

from src.config import (
    Config,
    FloatRange,
    MultitaskConfig,
    PretrainConfig,
    compose_config,
    load_config,
    load_multitask_config,
    load_pretrain_config,
    parse_search_space,
)

CONFIG_PATH = Path("configs/gpt2_medtext.yaml")


def test_load_config_reads_project_config() -> None:
    config: Config = load_config(CONFIG_PATH)

    assert config.seed == 42
    assert config.mlflow.tracking_uri == "sqlite:///mlflow.db"
    assert config.mlflow.experiment_name == "gpt2-medtext-fine-tuned"
    assert config.mlflow.registered_model_name == "gpt2-medtext-sft"
    assert config.mlflow.log_system_metrics is True
    assert config.model.model_id == "openai-community/gpt2"
    assert config.training.max_epochs == 20
    assert config.optuna.n_trials == 20


def test_load_config_parses_learning_rates_as_floats() -> None:
    config: Config = load_config(CONFIG_PATH)

    learning_rates: list[float] = config.grid_search.param_grid["learning_rate"]

    assert learning_rates == [1e-5, 2e-5, 5e-5]
    assert all(isinstance(rate, float) for rate in learning_rates)


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_parse_search_space_reads_ranges_and_choices() -> None:
    raw_space = {
        "learning_rate": {"low": 1e-5, "high": 1e-4, "log": True},
        "batch_size": [2, 8],
    }

    search_space = parse_search_space(raw_space)

    assert search_space["learning_rate"] == FloatRange(low=1e-5, high=1e-4, log=True)
    assert search_space["batch_size"] == [2, 8]


def test_parse_search_space_rejects_non_numeric_range() -> None:
    with pytest.raises(ValueError):
        parse_search_space({"learning_rate": {"low": "1e-5", "high": 1e-4}})


MODEL_FILES: list[Path] = sorted(Path("configs/model").glob("*.yaml"))


def test_compose_config_defaults_to_gpt2() -> None:
    raw = compose_config()

    assert raw["model"]["name"] == "gpt2"
    assert raw["results_dir"] == "results/gpt2"


@pytest.mark.parametrize("model_file", MODEL_FILES, ids=[path.stem for path in MODEL_FILES])
def test_every_model_file_fills_the_config_and_is_named_after_its_file(model_file: Path) -> None:
    # compose_config raises if a `???` value is left unset
    raw = compose_config([f"model={model_file.stem}"])

    assert raw["model"]["name"] == model_file.stem


def test_compose_config_rejects_an_unknown_model() -> None:
    with pytest.raises(Exception):
        compose_config(["model=not-a-model"])


def test_load_pretrain_config_names_mlflow_and_paths_after_the_model() -> None:
    config: PretrainConfig = load_pretrain_config(["model=qwen3-1.7b"])

    assert config.model.model_id == "Qwen/Qwen3-1.7B-Base"
    assert config.mlflow.experiment_name == "qwen3-1.7b-pubmed-pretrain"
    assert config.mlflow.registered_model_name == "qwen3-1.7b-pubmed-dapt"
    assert config.corpus.processed_dir == "data/processed/pubmed_v1_qwen3-1.7b"
    assert config.pretraining.output_dir == "checkpoints/pretrain/qwen3-1.7b-pubmed-dapt"
    assert config.results_dir == "results/qwen3-1.7b"


def test_load_pretrain_config_expands_the_corpus_directory() -> None:
    config: PretrainConfig = load_pretrain_config()

    assert not config.corpus.raw_dir.startswith("~")


def test_load_multitask_config_points_dapt_at_the_pretrained_model() -> None:
    config: MultitaskConfig = load_multitask_config(["model=qwen3-1.7b"])

    assert config.mlflow.experiment_name == "qwen3-1.7b-med-multitask"
    assert config.mlflow.registered_model_name == "qwen3-1.7b-med-multitask"
    assert config.finetune.dapt_model_id == "models:/qwen3-1.7b-pubmed-dapt@candidate"


def test_fine_tuning_data_is_the_same_for_every_model() -> None:
    gpt2: MultitaskConfig = load_multitask_config(["model=gpt2"])
    qwen: MultitaskConfig = load_multitask_config(["model=qwen3-1.7b"])

    assert gpt2.data_filter == qwen.data_filter
    assert gpt2.multitask == qwen.multitask
    assert gpt2.training.max_epochs == gpt2.multitask.mixture_epochs


def test_load_multitask_config_merges_the_model_search_range_into_the_sweep() -> None:
    sweep = load_multitask_config(["model=qwen3-1.7b"]).sweep

    assert sweep.search_space["learning_rate"] == FloatRange(low=5e-6, high=1e-4, log=True)
    assert sweep.search_space["batch_size"] == [8, 16, 32]
