"""Download the raw fine-tuning datasets from the Hugging Face Hub.

Every source in the `datasets` config section is downloaded at its pinned
revision and saved untouched as `<raw_dir>/<source>/<split>.parquet`, with a
`manifest.json` listing what was downloaded:

    uv run python -m src.download

Parquet keeps the original column types (nested PubMedQA contexts, MedQA
option maps), so the next step reads exactly what the Hub served.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from src.config import DatasetsConfig, SourceDataset, load_multitask_config


def download_source(source: SourceDataset, output_dir: Path) -> dict[str, int]:
    """Download one dataset at its pinned revision and save each split as Parquet.

    Args:
        source: Dataset id, subset and revision.
        output_dir: Directory for `<split>.parquet` (created if missing).

    Returns:
        Rows per split.
    """
    splits: DatasetDict = load_dataset(source.dataset_id, source.subset, revision=source.revision)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_parquet(output_dir / f"{name}.parquet")
    return {name: len(split) for name, split in splits.items()}


def download_datasets(datasets: DatasetsConfig) -> dict[str, Any] | None:
    """Download every source into `raw_dir` and write `manifest.json`.

    Args:
        datasets: Raw directory and the pinned sources.

    Returns:
        The manifest, or None if `raw_dir` already exists.
    """
    raw_dir = Path(datasets.raw_dir)
    if raw_dir.exists():
        print(f"{raw_dir} already exists")
        return None

    manifest: dict[str, Any] = {}
    for name, source in datasets.sources.items():
        print(f"Downloading {name}: {source.dataset_id} {source.subset or ''} @ {source.revision[:7]}")
        rows: dict[str, int] = download_source(source, raw_dir / name)
        manifest[name] = {
            "dataset_id": source.dataset_id,
            "subset": source.subset,
            "revision": source.revision,
            "rows": rows,
        }

    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def load_raw(source_dir: str | Path) -> DatasetDict:
    """Load the `<split>.parquet` files of one downloaded source.

    Args:
        source_dir: Directory written by `download_source`.

    Returns:
        Every split, named after its file.

    Raises:
        FileNotFoundError: If the directory has no Parquet files.
    """
    files: dict[str, str] = {path.stem: str(path) for path in sorted(Path(source_dir).glob("*.parquet"))}
    if not files:
        raise FileNotFoundError(f"No .parquet splits in {source_dir}")
    return load_dataset("parquet", data_files=files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the raw fine-tuning datasets.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args()

    download_datasets(load_multitask_config(args.overrides).datasets)


if __name__ == "__main__":
    main()
