"""The training mixture: how the unique examples are repeated and sampled in each epoch."""

import random
from collections import deque

from datasets import Dataset, DatasetDict, concatenate_datasets

from medllm.config import MixtureConfig


def balanced_sample_indices(labels: list[str], per_class: int, seed: int) -> list[int]:
    """Indices of up to `per_class` random items of each label."""
    indices_by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(index)

    rng = random.Random(seed)
    sample: list[int] = []
    for label in sorted(indices_by_label):
        indices: list[int] = indices_by_label[label]
        sample.extend(rng.sample(indices, min(per_class, len(indices))))
    return sorted(sample)


def epoch_chunks(num_items: int, per_epoch: int, num_epochs: int, seed: int) -> list[list[int]]:
    """Split shuffled passes over the items into per-epoch chunks.

    Every item is used once before any is repeated, and no chunk contains the
    same item twice.

    Raises:
        ValueError: If per_epoch is larger than num_items.
    """
    if per_epoch > num_items:
        raise ValueError(f"per_epoch ({per_epoch}) exceeds the number of items ({num_items})")

    rng = random.Random(seed)
    queue: deque[int] = deque()
    chunks: list[list[int]] = []

    for _ in range(num_epochs):
        chunk: list[int] = []
        chosen: set[int] = set()
        postponed: list[int] = []
        while len(chunk) < per_epoch:
            if not queue:
                order: list[int] = list(range(num_items))
                rng.shuffle(order)
                queue.extend(order)
            item: int = queue.popleft()
            if item in chosen:
                postponed.append(item)
                continue
            chunk.append(item)
            chosen.add(item)
        # Items already in this chunk are used first in the next one
        queue.extendleft(reversed(postponed))
        chunks.append(chunk)

    return chunks


def build_mixture(splits: DatasetDict, mixture: MixtureConfig, seed: int) -> Dataset:
    """Training mixture for `mixture.epochs` epochs (shuffled by the trainer).

    Per epoch: MedText, PubMedQA-labeled and MedQA repeated, PubMedQA-artificial
    balanced by decision (a new sample every epoch), and a new slice of MedMCQA.

    Args:
        splits: The unique training examples (`train_medtext`, `train_pubmedqa`,
            `train_pubmedqa_artificial`, `train_medmcqa`, `train_medqa`).
        mixture: Epochs, repeats and sample sizes.
        seed: Random seed.

    Returns:
        All epochs concatenated.
    """
    artificial: Dataset = splits["train_pubmedqa_artificial"]
    medmcqa: Dataset = splits["train_medmcqa"]
    artificial_labels: list[str] = artificial["label"]
    medmcqa_chunks: list[list[int]] = epoch_chunks(
        len(medmcqa), mixture.medmcqa_per_epoch, mixture.epochs, seed
    )

    parts: list[Dataset] = []
    for epoch in range(mixture.epochs):
        parts.extend([splits["train_medtext"]] * mixture.medtext_repeats)
        parts.extend([splits["train_pubmedqa"]] * mixture.pubmedqa_labeled_repeats)
        artificial_indices: list[int] = balanced_sample_indices(
            artificial_labels, mixture.pubmedqa_artificial_per_class, seed + epoch
        )
        parts.append(artificial.select(artificial_indices))
        parts.append(medmcqa.select(medmcqa_chunks[epoch]))
        parts.extend([splits["train_medqa"]] * mixture.medqa_repeats)

    return concatenate_datasets(parts)
