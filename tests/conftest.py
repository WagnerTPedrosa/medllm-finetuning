from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest

from medllm.config import MultitaskConfig, load_multitask_config


@pytest.fixture
def config() -> MultitaskConfig:
    return load_multitask_config()


@pytest.fixture
def tracking_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point MLflow at a throwaway SQLite store (artifacts land in tmp_path/mlruns)."""
    old_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "tracking-test")
    yield tmp_path
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)
