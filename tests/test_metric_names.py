import sqlite3
from pathlib import Path

from medllm.metric_names import (
    backup_database,
    metric_keys,
    metric_renames,
    rename_metrics,
)


def make_store(path: Path) -> None:
    """The three MLflow tables that hold metric names, with one run's metrics."""
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metrics (key TEXT, value FLOAT, timestamp INT, run_uuid TEXT, step INT, is_nan INT,
                PRIMARY KEY (key, timestamp, step, run_uuid, value, is_nan));
            CREATE TABLE latest_metrics (key TEXT, value FLOAT, timestamp INT, step INT, is_nan INT, run_uuid TEXT,
                PRIMARY KEY (key, run_uuid));
            CREATE TABLE logged_model_metrics (model_id TEXT, metric_name TEXT, metric_timestamp_ms INT,
                metric_step INT, metric_value FLOAT, experiment_id INT, run_id TEXT);
            INSERT INTO metrics VALUES ('loss', 2.0, 1, 'r1', 10, 0), ('loss', 1.8, 2, 'r1', 20, 0),
                ('eval_combined_loss', 1.6, 2, 'r1', 20, 0), ('system/cpu_utilization_percentage', 5, 1, 'r1', 0, 0);
            INSERT INTO latest_metrics VALUES ('loss', 1.8, 2, 20, 0, 'r1'), ('eval_combined_loss', 1.6, 2, 20, 0, 'r1');
            INSERT INTO logged_model_metrics VALUES ('m1', 'test_medqa_accuracy', 3, 0, 0.4, 1, 'r1');
            """
        )


def test_metric_renames_keeps_only_names_that_change() -> None:
    renames = metric_renames(["loss", "eval_combined_loss", "system/cpu_utilization_percentage", "custom"])

    assert renames == {"loss": "train/loss", "eval_combined_loss": "eval/combined_loss"}


def test_metric_keys_lists_every_name_in_the_store(tmp_path: Path) -> None:
    store = tmp_path / "mlflow.db"
    make_store(store)

    assert metric_keys(store) == [
        "eval_combined_loss",
        "loss",
        "system/cpu_utilization_percentage",
        "test_medqa_accuracy",
    ]


def test_rename_metrics_renames_history_latest_values_and_model_metrics(tmp_path: Path) -> None:
    store = tmp_path / "mlflow.db"
    make_store(store)

    updated = rename_metrics(store, metric_renames(metric_keys(store)))

    assert updated == {"metrics": 3, "latest_metrics": 2, "logged_model_metrics": 1}
    with sqlite3.connect(store) as connection:
        history = connection.execute("SELECT key, step, value FROM metrics ORDER BY key, step").fetchall()
        model_metrics = connection.execute("SELECT metric_name FROM logged_model_metrics").fetchall()
    assert history == [
        ("eval/combined_loss", 20, 1.6),
        ("system/cpu_utilization_percentage", 0, 5.0),
        ("train/loss", 10, 2.0),
        ("train/loss", 20, 1.8),
    ]
    assert model_metrics == [("test/medqa_accuracy",)]


def test_backup_database_copies_the_store_next_to_it(tmp_path: Path) -> None:
    store = tmp_path / "mlflow.db"
    make_store(store)

    backup = backup_database(store)

    assert backup.parent == tmp_path
    assert backup.name.startswith("mlflow.db.backup-")
    assert backup.read_bytes() == store.read_bytes()
