"""Rename the metrics of earlier MLflow runs to their sections (`loss` -> `train/loss`).

    uv run python -m medllm.metric_names mlflow.db            # show what would change
    uv run python -m medllm.metric_names mlflow.db --apply    # back up, then rename

New runs already log sectioned names (see `medllm.tracking.metric_section_name`);
this brings the older runs in line, so both appear in the same charts. MLflow
has no API to rename a metric, so the SQLite store is updated directly, after
a copy of it is saved next to it. Run it only while nothing is logging to the
store (no training or evaluation running).
"""

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from medllm.tracking import metric_section_name

# Table -> column holding the metric name
METRIC_NAME_COLUMNS: dict[str, str] = {
    "metrics": "key",
    "latest_metrics": "key",
    "logged_model_metrics": "metric_name",
}


def metric_renames(keys: list[str]) -> dict[str, str]:
    """Old name -> sectioned name, for the names that change."""
    renames: dict[str, str] = {}
    for key in keys:
        new_key: str = metric_section_name(key)
        if new_key != key:
            renames[key] = new_key
    return renames


def metric_keys(db_path: Path) -> list[str]:
    """Every metric name in the store, sorted."""
    keys: set[str] = set()
    with sqlite3.connect(db_path) as connection:
        for table, column in METRIC_NAME_COLUMNS.items():
            rows = connection.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()
            keys.update(row[0] for row in rows)
    return sorted(keys)


def rename_metrics(db_path: Path, renames: dict[str, str]) -> dict[str, int]:
    """Rename metrics in every table of the store, in one transaction.

    A row whose new name already exists with the same values is left under its
    old name (the update would duplicate it).

    Args:
        db_path: MLflow SQLite store.
        renames: Old name -> new name.

    Returns:
        Rows renamed per table.
    """
    updated: dict[str, int] = {}
    with sqlite3.connect(db_path) as connection:
        for table, column in METRIC_NAME_COLUMNS.items():
            count = 0
            for old_key, new_key in renames.items():
                cursor = connection.execute(
                    f"UPDATE OR IGNORE {table} SET {column} = ? WHERE {column} = ?", (new_key, old_key)
                )
                count += cursor.rowcount
            updated[table] = count
    return updated


def backup_database(db_path: Path) -> Path:
    """Copy the store to `<name>.backup-<timestamp>` in the same folder and return its path."""
    stamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup: Path = db_path.with_name(f"{db_path.name}.backup-{stamp}")
    shutil.copy2(db_path, backup)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename earlier MLflow metrics to their sections.")
    parser.add_argument("db_path", type=Path, help="MLflow SQLite store, e.g. mlflow.db")
    parser.add_argument("--apply", action="store_true", help="Back up the store and rename (default: only show)")
    args = parser.parse_args()

    renames: dict[str, str] = metric_renames(metric_keys(args.db_path))
    for old_key, new_key in renames.items():
        print(f"{old_key} -> {new_key}")
    print(f"{len(renames)} metric names to rename")
    if not args.apply:
        print("Nothing changed; run again with --apply to rename.")
        return

    print(f"Backup: {backup_database(args.db_path)}")
    print(f"Rows renamed: {rename_metrics(args.db_path, renames)}")


if __name__ == "__main__":
    main()
