import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from datasets import load_from_disk

DATASET_PATH = "data/processed/medtext_v1"
DATASET_ID = "BI55/MedText"
OUTPUT_DIR = Path("data/processed/medtext_v1")
SEED = 42


def prepare_dataset():
    # 1. Identify and pin the current dataset revision
    revision = HfApi().dataset_info(DATASET_ID).sha

    # 2. Load raw data
    raw = load_dataset(
        DATASET_ID,
        split="train",
        revision=revision,
    )

    # 3. Validate schema
    required_columns = {"Prompt", "Completion"}

    if not required_columns.issubset(raw.column_names):
        raise ValueError(
            f"Missing required columns: " f"{required_columns - set(raw.column_names)}"
        )

    # 4. Clean and deduplicate
    df = raw.select_columns(["Prompt", "Completion"]).to_pandas()

    for column in ["Prompt", "Completion"]:
        df[column] = df[column].astype("string").str.strip()

    df = df.dropna(subset=["Prompt", "Completion"])

    df = df[(df["Prompt"] != "") & (df["Completion"] != "")]

    df = df.drop_duplicates(subset=["Prompt"])

    df = df.rename(
        columns={
            "Prompt": "prompt",
            "Completion": "completion",
        }
    )

    dataset = Dataset.from_pandas(
        df,
        preserve_index=False,
    )

    # 5. Create reproducible 80/10/10 splits
    split = dataset.train_test_split(
        test_size=0.2,
        seed=SEED,
    )

    holdout = split["test"].train_test_split(
        test_size=0.5,
        seed=SEED,
    )

    datasets = DatasetDict(
        {
            "train": split["train"],
            "validation": holdout["train"],
            "test": holdout["test"],
        }
    )

    # 6. Save processed dataset
    if OUTPUT_DIR.exists():
        print(f"{OUTPUT_DIR} already exists")
        return

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    datasets.save_to_disk(str(OUTPUT_DIR))

    print(datasets)
    print(f"Dataset saved to {OUTPUT_DIR}")


def load_medtext():
    dataset = load_from_disk(DATASET_PATH)
    return dataset["train"], dataset["validation"], dataset["test"]
