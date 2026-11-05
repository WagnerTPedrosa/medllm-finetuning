
from datasets import Dataset, DatasetDict

from medllm.config import MixtureConfig
from medllm.data.mixture import balanced_sample_indices, build_mixture, epoch_chunks


def task_rows(task: str, count: int, labels: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [f"{task} {i}" for i in range(count)],
            "completion": ["x"] * count,
            "task": [task] * count,
            "label": [labels[i % len(labels)] for i in range(count)],
        }
    )


def test_balanced_sample_indices_caps_each_label() -> None:
    labels = ["yes"] * 90 + ["no"] * 10

    indices = balanced_sample_indices(labels, per_class=20, seed=0)

    picked = [labels[i] for i in indices]
    assert (picked.count("yes"), picked.count("no")) == (20, 10)


def test_epoch_chunks_uses_every_item_before_repeating() -> None:
    chunks = epoch_chunks(num_items=10, per_epoch=4, num_epochs=3, seed=0)

    assert [len(chunk) for chunk in chunks] == [4, 4, 4]
    first_pass = chunks[0] + chunks[1] + chunks[2][:2]
    assert sorted(first_pass) == list(range(10))
    assert all(len(set(chunk)) == len(chunk) for chunk in chunks)


def test_build_mixture_has_the_configured_examples_per_epoch() -> None:
    mixture_config = MixtureConfig(
        epochs=2,
        medtext_repeats=2,
        pubmedqa_labeled_repeats=3,
        pubmedqa_artificial_per_class=5,
        medmcqa_per_epoch=7,
        medqa_repeats=2,
    )
    splits = DatasetDict(
        {
            "train_medtext": task_rows("medtext", 4, [""]),
            "train_pubmedqa": task_rows("pubmedqa", 2, ["yes"]),
            "train_pubmedqa_artificial": task_rows("pubmedqa_artificial", 30, ["yes", "yes", "no"]),
            "train_medmcqa": task_rows("medmcqa", 10, ["A"]),
            "train_medqa": task_rows("medqa", 3, ["B"]),
        }
    )

    mixture = build_mixture(splits, mixture_config, seed=0)

    tasks = mixture["task"]
    assert tasks.count("medtext") == 2 * 4 * 2
    assert tasks.count("pubmedqa") == 2 * 2 * 3
    assert tasks.count("pubmedqa_artificial") == 2 * (5 + 5)
    assert tasks.count("medmcqa") == 2 * 7
    assert tasks.count("medqa") == 2 * 3 * 2
    assert len(mixture) == 16 + 12 + 20 + 14 + 12
