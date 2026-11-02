from pathlib import Path

import pytest

from src.config import Config, FloatRange, load_config, parse_search_space

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
