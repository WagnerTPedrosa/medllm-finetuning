"""Multitask fine-tuning (MedText, PubMedQA, MedMCQA, MedQA), with LoRA or all weights.

    uv run python -m medllm.training.finetune --base original
    uv run python -m medllm.training.finetune --base dapt finetune.method=full

`--base original` fine-tunes the Hub model, `--base dapt` the PubMed-pretrained
one (`finetune.dapt_model_id`), with LoRA or full fine-tuning
(`finetune.method`). Results go to `results/<model>/finetune_<method>_<base>.json`,
with each test answer marked right or wrong for the confidence intervals of
the report.

Each seed in `finetune.seeds` trains one model as a nested MLflow run under a
`multitask-<method>-<base>` run. The training set is the mixture of
`mixture.epochs` epochs built from the unique `train_*` examples; the model is
evaluated on the combined validation set after every mixture epoch, early
stopping keeps the best one (LoRA adapters are merged into the model), and
that model is scored on the four test sets.
"""

import argparse
import json
import math
import statistics
from dataclasses import asdict
from operator import itemgetter
from pathlib import Path
from typing import Any

import mlflow
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, EarlyStoppingCallback, PreTrainedModel
from trl import SFTConfig, SFTTrainer

from medllm.config import Hyperparameters, MultitaskConfig, load_multitask_config
from medllm.data.finetune_data import load_finetune_data
from medllm.data.mixture import build_mixture
from medllm.data.preprocessing import SFT_COLUMNS
from medllm.evaluation.tasks import TaskEvaluation, evaluate_on_tasks
from medllm.modeling import (
    build_peft_config,
    load_causal_lm,
    merge_lora,
    model_label,
    resolve_model_source,
    set_end_of_turn,
    weights_dtype,
)
from medllm.reporting.report import BASES, save_stage_result
from medllm.tracking import configure_tracking, log_best_model, start_stage_run

VALIDATION_SETS = ("combined", "medtext", "pubmedqa", "medmcqa", "medqa")


LOGGING_STEPS = 500


def steps_per_epoch(examples_per_epoch: int, batch_size: int) -> int:
    """Optimizer steps in one mixture epoch (single device, no accumulation)."""
    return math.ceil(examples_per_epoch / batch_size)


def default_hyperparameters(config: MultitaskConfig) -> Hyperparameters:
    """Hyperparameters of the full runs: `finetune` plus the schedule from `training`."""
    return {
        "learning_rate": config.finetune.learning_rate,
        "batch_size": config.finetune.batch_size,
        "weight_decay": config.training.weight_decay,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "warmup_ratio": config.training.warmup_ratio,
    }


def build_multitask_sft_config(
    config: MultitaskConfig,
    output_dir: Path,
    seed: int,
    epoch_steps: int,
    hyperparameters: Hyperparameters | None = None,
    save_models: bool = True,
) -> SFTConfig:
    """SFTTrainer arguments: one pass over the multi-epoch mixture, evaluated every `epoch_steps`.

    Args:
        config: Multitask configuration.
        output_dir: Checkpoint directory for this run.
        seed: Random seed of this run.
        epoch_steps: Optimizer steps between evaluations (one mixture epoch in
            full runs, a fraction of it in sweep trials).
        hyperparameters: Values that override `default_hyperparameters`
            (sweep trials).
        save_models: Keep checkpoints and reload the best one at the end;
            sweep trials only need the validation losses.

    Returns:
        The SFTTrainer configuration (model selection on `eval_combined_loss`).
    """
    values: Hyperparameters = {**default_hyperparameters(config), **(hyperparameters or {})}
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        learning_rate=values["learning_rate"],
        per_device_train_batch_size=values["batch_size"],
        per_device_eval_batch_size=config.training.eval_batch_size,
        weight_decay=values["weight_decay"],
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=values["lr_scheduler_type"],
        # A float in [0, 1) is a fraction of the total training steps
        warmup_steps=values["warmup_ratio"],
        completion_only_loss=True,
        max_length=config.model.max_seq_len,
        eval_strategy="steps",
        eval_steps=epoch_steps,
        save_strategy="steps" if save_models else "no",
        save_steps=epoch_steps,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=save_models,
        metric_for_best_model="eval_combined_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.finetune.gradient_checkpointing,
        # The non-reentrant variant also works when only LoRA adapters need gradients
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        report_to="mlflow",
    )


def epoch_metrics(log_history: list[dict[str, float]], validation_sets: list[str]) -> dict[str, list[float]]:
    """Validation loss after every evaluation, per validation set (e.g. `{"combined": [2.2, 2.0]}`)."""
    return {
        name: [log[f"eval_{name}_loss"] for log in log_history if f"eval_{name}_loss" in log]
        for name in validation_sets
    }


def summarize_seeds(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation of every numeric metric across seeds."""
    numeric_keys: list[str] = [
        key for key, value in results[0].items() if isinstance(value, (int, float))
    ]
    return {
        key: {
            "mean": statistics.mean(result[key] for result in results),
            "std": statistics.stdev(result[key] for result in results) if len(results) > 1 else 0.0,
        }
        for key in numeric_keys
    }


def finetune_stage(method: str, base: str) -> str:
    """Name of the stage result file (`finetune_lora_original`)."""
    return f"finetune_{method}_{base}"


def base_model_id(config: MultitaskConfig, base: str) -> str:
    """Model to fine-tune: the Hub model (`original`) or the PubMed-pretrained one (`dapt`).

    Raises:
        ValueError: If `base` is not one of `BASES`.
    """
    if base == "original":
        return config.model.model_id
    if base == "dapt":
        return config.finetune.dapt_model_id
    raise ValueError(f"base must be one of {BASES}, got {base!r}")


def train_multitask(
    config: MultitaskConfig, model_id: str, seed: int, splits: DatasetDict
) -> dict[str, Any]:
    """Fine-tune one seed as a nested run, then score the best epoch on the test sets.

    Args:
        config: Multitask configuration.
        model_id: Base model (Hub id, path or MLflow URI).
        seed: Random seed.
        splits: Splits written by `medllm.data.finetune_data filter_length`.

    Returns:
        Run id, model directory, number of mixture epochs trained, best validation
        loss and the test metrics.
    """
    train_set: Dataset = build_mixture(splits, config.mixture, config.seed)
    examples_per_epoch: int = len(train_set) // config.mixture.epochs
    epoch_steps: int = steps_per_epoch(examples_per_epoch, config.finetune.batch_size)
    method: str = config.finetune.method
    output_dir = Path(config.finetune.output_dir) / method / f"{model_label(model_id)}-seed{seed}"
    source: str = resolve_model_source(model_id)

    with start_stage_run(
        "finetune",
        f"multitask-{method}-{model_label(model_id)}-seed{seed}",
        config,
        {"task": "multitask", "finetune_seed": str(seed), "base_model": model_id, "finetune_method": method},
        nested=mlflow.active_run() is not None,
    ) as run:
        mlflow.log_params({"examples_per_epoch": examples_per_epoch, "epoch_steps": epoch_steps})
        trainer = SFTTrainer(
            model=load_causal_lm(source, weights_dtype(method)),
            args=build_multitask_sft_config(config, output_dir, seed, epoch_steps),
            train_dataset=train_set.select_columns(SFT_COLUMNS),
            eval_dataset={name: splits[f"validation_{name}"].select_columns(SFT_COLUMNS) for name in VALIDATION_SETS},
            processing_class=AutoTokenizer.from_pretrained(source),
            callbacks=[EarlyStoppingCallback(config.training.early_stopping_patience)],
            peft_config=build_peft_config(config),
        )
        trainer.train()

        curves: dict[str, list[float]] = epoch_metrics(trainer.state.log_history, list(VALIDATION_SETS))
        best_dir: Path = output_dir / "best"
        model: PreTrainedModel = merge_lora(trainer.model)
        set_end_of_turn(model, trainer.processing_class)
        model.save_pretrained(str(best_dir))
        trainer.processing_class.save_pretrained(str(best_dir))
        evaluation: TaskEvaluation = evaluate_on_tasks(model, trainer.processing_class, splits, config, "test")
        summary: dict[str, float] = {
            "best_validation_combined_loss": min(curves["combined"]),
            "mixture_epochs_trained": len(curves["combined"]),
            **evaluation.metrics,
        }
        mlflow.log_metrics(summary)

    return {"run_id": run.info.run_id, "model_dir": str(best_dir), **summary, "correct": evaluation.correct}


def run_multitask_seeds(config: MultitaskConfig, base: str) -> dict[str, Any]:
    """Fine-tune every seed in `finetune.seeds` from one starting point and summarize them.

    Logs a `multitask-<method>-<model>` MLflow run with the configuration, the
    mean and standard deviation of every metric, and one nested run per seed.
    The seed with the lowest validation loss is registered as
    `models:/<registered_model_name>@<method>-<base>-best`.

    Args:
        config: Multitask configuration.
        base: `original` or `dapt` (see `base_model_id`).

    Returns:
        The starting model, the parent run id, the registered model, per-seed
        results (with each test answer marked right or wrong) and their summary.
    """
    model_id: str = base_model_id(config, base)
    method: str = config.finetune.method
    splits: DatasetDict = load_finetune_data(config.multitask.processed_dir)
    configure_tracking(config)

    with start_stage_run(
        "finetune",
        f"multitask-{method}-{model_label(model_id)}",
        config,
        {"task": "multitask", "base_model": model_id, "finetune_base": base, "finetune_method": method},
    ) as parent:
        mlflow.log_dict(asdict(config), "config.json")
        mlflow.log_artifact(str(Path(config.multitask.processed_dir) / "stats.json"), "data")
        results: list[dict[str, Any]] = [
            train_multitask(config, model_id, seed, splits) for seed in config.finetune.seeds
        ]
        correct_by_seed: list[dict[str, list[int]]] = [result.pop("correct") for result in results]
        summary = summarize_seeds(results)
        mlflow.log_metrics({f"{key}_mean": value["mean"] for key, value in summary.items()})
        mlflow.log_metrics({f"{key}_std": value["std"] for key, value in summary.items()})
        mlflow.log_dict({"results": results, "summary": summary}, "seed_results.json")

        best = min(results, key=itemgetter("best_validation_combined_loss"))
        alias = f"{method}-{base}-best"
        log_best_model(best["run_id"], best["model_dir"], config.mlflow.registered_model_name, alias)

    return {
        "base": base,
        "method": method,
        "model_id": model_id,
        "parent_run_id": parent.info.run_id,
        "registered_model_uri": f"models:/{config.mlflow.registered_model_name}@{alias}",
        "seeds": [
            {"seed": seed, "run_id": result["run_id"], "model_dir": result["model_dir"], "correct": correct}
            for seed, result, correct in zip(config.finetune.seeds, results, correct_by_seed)
        ],
        "results": results,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multitask fine-tuning on MedText, PubMedQA, MedMCQA and MedQA.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example finetune.method=full")
    parser.add_argument("--base", choices=BASES, default="original", help="Model to fine-tune")
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    outcome: dict[str, Any] = run_multitask_seeds(config, args.base)
    save_stage_result(config.results_dir, finetune_stage(config.finetune.method, args.base), outcome)
    print(json.dumps(outcome["summary"], indent=2))


if __name__ == "__main__":
    main()
