from pathlib import Path

import mlflow
import pytest

from medllm.mlflow_migration import (
    experiment_renames,
    legacy_run_variant,
    migrate_runs,
    registered_model_renames,
)


def test_experiment_renames_move_a_model_into_the_shared_experiments() -> None:
    assert experiment_renames("qwen3-1.7b") == {
        "qwen3-1.7b-pubmed-pretrain": "pubmed-dapt",
        "qwen3-1.7b-med-multitask": "med-qa-selection",
    }


def test_registered_model_renames_drop_multitask() -> None:
    assert registered_model_renames("qwen3-1.7b") == {"qwen3-1.7b-med-multitask": "qwen3-1.7b-med-qa"}


@pytest.mark.parametrize(
    ("tags", "parent_tags", "expected"),
    [
        # Zero-shot runs from before the DAPT branch have no base: the Hub model
        ({"stage": "evaluate", "task": "multitask"}, {}, "base"),
        ({"stage": "evaluate", "task": "multitask", "finetune_base": "dapt"}, {}, "dapt"),
        ({"stage": "finetune", "task": "multitask", "finetune_base": "original", "finetune_method": "lora"}, {}, "lora"),
        # Seed runs take the base from their parent
        (
            {"stage": "finetune", "task": "multitask", "finetune_method": "lora"},
            {"finetune_base": "dapt"},
            "dapt-lora",
        ),
        # Pretraining and sweep runs have no variant
        ({"stage": "pretrain", "run_type": "full"}, {}, None),
        ({"stage": "finetune", "task": "multitask_sweep"}, {}, None),
    ],
)
def test_legacy_run_variant_reads_the_recipe_from_old_tags(
    tags: dict[str, str], parent_tags: dict[str, str], expected: str | None
) -> None:
    assert legacy_run_variant(tags, parent_tags) == expected


def make_legacy_store(client: mlflow.MlflowClient) -> dict[str, str]:
    """Experiments, runs and a registered model named as before the shared experiments."""
    pretrain_id = client.create_experiment("qwen3-1.7b-pubmed-pretrain")
    selection_id = client.create_experiment("qwen3-1.7b-med-multitask")
    pretrain = client.create_run(pretrain_id, tags={"stage": "pretrain"})
    zero_shot = client.create_run(selection_id, tags={"stage": "evaluate", "task": "multitask"})
    parent = client.create_run(
        selection_id,
        tags={"stage": "finetune", "task": "multitask", "finetune_base": "original", "finetune_method": "lora"},
    )
    seed = client.create_run(
        selection_id,
        tags={"stage": "finetune", "task": "multitask", "finetune_method": "lora", "mlflow.parentRunId": parent.info.run_id},
    )
    client.create_registered_model("qwen3-1.7b-med-multitask")
    return {
        "pretrain": pretrain.info.run_id,
        "zero_shot": zero_shot.info.run_id,
        "parent": parent.info.run_id,
        "seed": seed.info.run_id,
    }


def test_migrate_runs_only_reports_without_apply(tracking_store: Path) -> None:
    client = mlflow.MlflowClient(f"sqlite:///{tracking_store / 'tracking.db'}")
    make_legacy_store(client)

    actions = migrate_runs(client, "qwen3-1.7b", apply=False)

    assert "rename experiment qwen3-1.7b-med-multitask -> med-qa-selection" in actions
    assert client.get_experiment_by_name("med-qa-selection") is None


def test_migrate_runs_renames_and_tags_with_apply(tracking_store: Path) -> None:
    client = mlflow.MlflowClient(f"sqlite:///{tracking_store / 'tracking.db'}")
    runs = make_legacy_store(client)

    migrate_runs(client, "qwen3-1.7b", apply=True)

    assert client.get_experiment_by_name("pubmed-dapt") is not None
    assert client.get_experiment_by_name("med-qa-selection") is not None
    assert client.get_registered_model("qwen3-1.7b-med-qa").name == "qwen3-1.7b-med-qa"
    tags = {name: client.get_run(run_id).data.tags for name, run_id in runs.items()}
    assert {name: run_tags["model"] for name, run_tags in tags.items()} == dict.fromkeys(runs, "qwen3-1.7b")
    assert "variant" not in tags["pretrain"]
    assert tags["zero_shot"]["variant"] == "base"
    assert tags["parent"]["variant"] == "lora"
    assert tags["seed"]["variant"] == "lora"
    assert tags["seed"]["finetune_base"] == "original"


def test_migrate_runs_refuses_to_merge_into_an_existing_experiment(tracking_store: Path) -> None:
    client = mlflow.MlflowClient(f"sqlite:///{tracking_store / 'tracking.db'}")
    make_legacy_store(client)
    client.create_experiment("med-qa-selection")

    with pytest.raises(ValueError, match="med-qa-selection"):
        migrate_runs(client, "qwen3-1.7b", apply=True)
