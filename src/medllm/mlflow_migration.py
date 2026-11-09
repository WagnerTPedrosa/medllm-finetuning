"""Move the MLflow runs of one model into the shared experiments, and tag them.

    uv run python -m medllm.mlflow_migration model=qwen3-1.7b            # show what would change
    uv run python -m medllm.mlflow_migration model=qwen3-1.7b --apply    # change it

Runs used to go to one experiment per model (`qwen3-1.7b-med-multitask`);
now every model shares `pubmed-dapt` and `med-qa-selection`, told apart by the
`model` and `variant` tags (see `medllm.tracking`). This renames the old
experiments and the old registered model (aliases are kept) and adds the
two tags to the old runs. Run `medllm.metric_names` too, for the metric names.

A model whose experiments are renamed first owns the shared experiments: a
second model (another machine's store, say) cannot be merged into them here,
because MLflow cannot move runs between experiments.
"""

import argparse

import mlflow
from mlflow.entities import Run

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.tracking import run_variant

PARENT_RUN_TAG = "mlflow.parentRunId"


def experiment_renames(model_name: str) -> dict[str, str]:
    """Old experiment of the model -> shared experiment."""
    return {
        f"{model_name}-pubmed-pretrain": "pubmed-dapt",
        f"{model_name}-med-multitask": "med-qa-selection",
    }


def registered_model_renames(model_name: str) -> dict[str, str]:
    """Old registered model -> new name."""
    return {f"{model_name}-med-multitask": f"{model_name}-med-qa"}


def legacy_run_base(tags: dict[str, str], parent_tags: dict[str, str]) -> str:
    """Base of an old run: its own tag, its parent's, or `original` (before the DAPT branch)."""
    return tags.get("finetune_base", parent_tags.get("finetune_base", "original"))


def legacy_run_variant(tags: dict[str, str], parent_tags: dict[str, str]) -> str | None:
    """Variant of a run logged before the `variant` tag, from its other tags.

    Args:
        tags: The run's tags.
        parent_tags: Tags of its parent run (empty for a top-level run).

    Returns:
        The variant (see `run_variant`), or None for pretraining and sweep runs.
    """
    if tags.get("task") != "multitask":
        return None
    base: str = legacy_run_base(tags, parent_tags)
    if tags.get("stage") == "evaluate":
        return run_variant(base, None)
    if tags.get("stage") == "finetune" and "finetune_method" in tags:
        return run_variant(base, tags["finetune_method"])
    return None


def check_free_names(client: mlflow.MlflowClient, renames: dict[str, str]) -> None:
    """Raise ValueError if a shared experiment already exists while the old one still does."""
    for old_name, new_name in renames.items():
        if client.get_experiment_by_name(old_name) is None:
            continue
        if client.get_experiment_by_name(new_name) is not None:
            raise ValueError(
                f"Experiment {new_name!r} already exists: runs of {old_name!r} cannot be merged into it"
            )


def run_tag_updates(run: Run, parent: Run | None, model_name: str) -> dict[str, str]:
    """Tags to add to an old run: `model`, and `variant` (plus `finetune_base` for seed runs)."""
    tags: dict[str, str] = run.data.tags
    parent_tags: dict[str, str] = parent.data.tags if parent is not None else {}
    updates: dict[str, str] = {}
    if "model" not in tags:
        updates["model"] = model_name
    variant: str | None = legacy_run_variant(tags, parent_tags)
    if variant is not None and "variant" not in tags:
        updates["variant"] = variant
        if "finetune_base" not in tags and tags.get("stage") == "finetune":
            updates["finetune_base"] = legacy_run_base(tags, parent_tags)
    return updates


def migrate_runs(client: mlflow.MlflowClient, model_name: str, apply: bool) -> list[str]:
    """Rename the model's experiments and registered model, and tag its runs.

    Args:
        client: Client of the tracking store.
        model_name: Model whose runs to migrate (`qwen3-1.7b`).
        apply: Make the changes; otherwise only list them.

    Returns:
        One line per change.

    Raises:
        ValueError: If a shared experiment already exists (checked before any change).
    """
    renames: dict[str, str] = experiment_renames(model_name)
    check_free_names(client, renames)
    actions: list[str] = []

    experiment_ids: list[str] = []
    for old_name, new_name in renames.items():
        experiment = client.get_experiment_by_name(old_name)
        if experiment is None:
            continue
        experiment_ids.append(experiment.experiment_id)
        actions.append(f"rename experiment {old_name} -> {new_name}")
        if apply:
            client.rename_experiment(experiment.experiment_id, new_name)

    runs: list[Run] = client.search_runs(experiment_ids) if experiment_ids else []
    runs_by_id: dict[str, Run] = {run.info.run_id: run for run in runs}
    for run in runs:
        parent: Run | None = runs_by_id.get(run.data.tags.get(PARENT_RUN_TAG, ""))
        for key, value in run_tag_updates(run, parent, model_name).items():
            actions.append(f"tag run {run.info.run_id} ({run.info.run_name}): {key}={value}")
            if apply:
                client.set_tag(run.info.run_id, key, value)

    for old_name, new_name in registered_model_renames(model_name).items():
        if not client.search_registered_models(filter_string=f"name = '{old_name}'"):
            continue
        actions.append(f"rename registered model {old_name} -> {new_name}")
        if apply:
            client.rename_registered_model(old_name, new_name)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Move a model's MLflow runs into the shared experiments.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    parser.add_argument("--apply", action="store_true", help="Make the changes (default: only show them)")
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    client = mlflow.MlflowClient(config.mlflow.tracking_uri)
    for action in migrate_runs(client, config.model.name, args.apply):
        print(action)
    if not args.apply:
        print("Nothing changed; run again with --apply to make these changes.")


if __name__ == "__main__":
    main()
