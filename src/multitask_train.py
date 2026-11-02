"""Multitask fine-tuning (MedText + PubMedQA + MedMCQA) and its evaluation.

Run from the repository root:

    uv run python -m src.multitask_train --config configs/gpt2_multitask.yaml
    uv run python -m src.multitask_train --model-id models:/gpt2-pubmed-dapt@champion
    uv run python -m src.multitask_train --sweep

Each seed in `finetune.seeds` trains one model as a nested MLflow run under a
`multitask-<base>` run. The training set holds `training.max_epochs` mixture
epochs; the model is evaluated on the combined dev set after every mixture
epoch, early stopping keeps the best one, and that model is scored on the
three test sets.
"""

import argparse
import json
import math
import shutil
import statistics
from dataclasses import asdict
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any

import mlflow
import optuna
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from src.choice_eval import evaluate_choice_task
from src.config import Hyperparameters, MultitaskConfig, load_multitask_config
from src.evaluation import evaluate
from src.modeling import get_device, resolve_model_source
from src.multitask import MEDMCQA_LETTERS, PUBMEDQA_DECISIONS, load_multitask
from src.tracking import configure_tracking, log_best_model, start_stage_run
from src.train import OptunaPruningCallback, free_gpu_memory, suggest_hyperparameters

DEV_SETS = ("combined", "medtext", "pubmedqa", "medmcqa")
SFT_COLUMNS = ["prompt", "completion"]
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
            sweep trials only need the dev losses.

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
        seed=seed,
        report_to="mlflow",
    )


def epoch_metrics(log_history: list[dict[str, float]], dev_sets: list[str]) -> dict[str, list[float]]:
    """Dev loss after every evaluation, per dev set (e.g. `{"combined": [2.2, 2.0]}`)."""
    return {
        name: [log[f"eval_{name}_loss"] for log in log_history if f"eval_{name}_loss" in log]
        for name in dev_sets
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


def evaluate_on_tasks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    splits: DatasetDict,
    config: MultitaskConfig,
    prefix: str,
) -> dict[str, float]:
    """Score a model on the MedText, PubMedQA and MedMCQA splits with a prefix.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer matching the model.
        splits: Splits written by `prepare_multitask`.
        config: Multitask configuration.
        prefix: `test` (final results) or `dev` (model selection, sweeps).

    Returns:
        `<prefix>_medtext_loss`, and accuracy and macro F1 for PubMedQA and MedMCQA.
    """
    device: torch.device = next(model.parameters()).device
    medtext: Dataset = splits[f"{prefix}_medtext"].select_columns(SFT_COLUMNS)
    pubmedqa = evaluate_choice_task(
        model, tokenizer, splits[f"{prefix}_pubmedqa"], list(PUBMEDQA_DECISIONS), device
    )
    medmcqa = evaluate_choice_task(
        model, tokenizer, splits[f"{prefix}_medmcqa"], list(MEDMCQA_LETTERS), device
    )
    return {
        f"{prefix}_medtext_loss": evaluate(model, tokenizer, medtext, config),
        f"{prefix}_pubmedqa_accuracy": pubmedqa["accuracy"],
        f"{prefix}_pubmedqa_macro_f1": pubmedqa["macro_f1"],
        f"{prefix}_medmcqa_accuracy": medmcqa["accuracy"],
        f"{prefix}_medmcqa_macro_f1": medmcqa["macro_f1"],
    }


def evaluate_model(model_id: str, config: MultitaskConfig, prefix: str = "test") -> dict[str, float]:
    """Load a model (Hub id, path or MLflow URI) and score it with `evaluate_on_tasks`."""
    source: str = resolve_model_source(model_id)
    model = AutoModelForCausalLM.from_pretrained(source).to(get_device())
    tokenizer = AutoTokenizer.from_pretrained(source)
    return evaluate_on_tasks(model, tokenizer, load_multitask(config.multitask.processed_dir), config, prefix)


def model_label(model_id: str) -> str:
    """Short name for directories and run names (`gpt2`, `gpt2-pubmed-dapt@champion`)."""
    return model_id.rstrip("/").split("/")[-1]


def train_multitask(
    config: MultitaskConfig, model_id: str, seed: int, splits: DatasetDict
) -> dict[str, Any]:
    """Fine-tune one seed as a nested run, then score the best epoch on the test sets.

    Args:
        config: Multitask configuration.
        model_id: Base model (Hub id, path or MLflow URI).
        seed: Random seed.
        splits: Splits written by `prepare_multitask`.

    Returns:
        Run id, model directory, number of mixture epochs trained, best dev
        loss and the test metrics.
    """
    stats: dict[str, Any] = json.loads(
        (Path(config.multitask.processed_dir) / "stats.json").read_text()
    )
    epoch_steps: int = steps_per_epoch(stats["examples_per_epoch"], config.finetune.batch_size)
    output_dir = Path(config.finetune.output_dir) / f"{model_label(model_id)}-seed{seed}"
    source: str = resolve_model_source(model_id)

    with start_stage_run(
        "finetune",
        f"multitask-{model_label(model_id)}-seed{seed}",
        config,
        {"task": "multitask", "finetune_seed": str(seed), "base_model": model_id},
        nested=mlflow.active_run() is not None,
    ) as run:
        mlflow.log_params({"examples_per_epoch": stats["examples_per_epoch"], "epoch_steps": epoch_steps})
        trainer = SFTTrainer(
            model=AutoModelForCausalLM.from_pretrained(source),
            args=build_multitask_sft_config(config, output_dir, seed, epoch_steps),
            train_dataset=splits["train"].select_columns(SFT_COLUMNS),
            eval_dataset={name: splits[f"dev_{name}"].select_columns(SFT_COLUMNS) for name in DEV_SETS},
            processing_class=AutoTokenizer.from_pretrained(source),
            callbacks=[EarlyStoppingCallback(config.training.early_stopping_patience)],
        )
        trainer.train()

        curves: dict[str, list[float]] = epoch_metrics(trainer.state.log_history, list(DEV_SETS))
        best_dir: Path = output_dir / "best"
        trainer.save_model(str(best_dir))
        test_metrics: dict[str, float] = evaluate_on_tasks(
            trainer.model, trainer.processing_class, splits, config, "test"
        )
        summary: dict[str, float] = {
            "best_dev_combined_loss": min(curves["combined"]),
            "mixture_epochs_trained": len(curves["combined"]),
            **test_metrics,
        }
        mlflow.log_metrics(summary)

    return {"run_id": run.info.run_id, "model_dir": str(best_dir), **summary}


def run_multitask_seeds(config: MultitaskConfig, model_id: str | None = None) -> dict[str, Any]:
    """Fine-tune every seed in `finetune.seeds` and summarize them.

    Logs a `multitask-<base>` MLflow run with the configuration, the mean and
    standard deviation of every metric, and one nested run per seed. The seed
    with the lowest dev loss is registered as
    `models:/<registered_model_name>@<base>-best`.

    Args:
        config: Multitask configuration.
        model_id: Base model; defaults to `config.model.model_id`.

    Returns:
        Per-seed results and their summary.
    """
    base: str = model_id or config.model.model_id
    splits: DatasetDict = load_multitask(config.multitask.processed_dir)
    configure_tracking(config)

    with start_stage_run(
        "finetune", f"multitask-{model_label(base)}", config, {"task": "multitask", "base_model": base}
    ):
        mlflow.log_dict(asdict(config), "config.json")
        mlflow.log_artifact(str(Path(config.multitask.processed_dir) / "stats.json"), "data")
        results: list[dict[str, Any]] = [
            train_multitask(config, base, seed, splits) for seed in config.finetune.seeds
        ]
        summary = summarize_seeds(results)
        mlflow.log_metrics({f"{key}_mean": value["mean"] for key, value in summary.items()})
        mlflow.log_metrics({f"{key}_std": value["std"] for key, value in summary.items()})
        mlflow.log_dict({"results": results, "summary": summary}, "seed_results.json")

        best = min(results, key=itemgetter("best_dev_combined_loss"))
        log_best_model(
            best["run_id"],
            best["model_dir"],
            config.mlflow.registered_model_name,
            alias=f"{model_label(base).replace('@', '-')}-best",
        )

    return {"results": results, "summary": summary}


class StopAfterStepsCallback(TrainerCallback):
    """End training after `max_steps` optimizer steps.

    Sweep trials keep the learning-rate schedule of a full run (its length
    comes from the whole mixture) but stop after their budget, so the first
    epoch of a trial matches the first epoch of the final run.
    """

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        if state.global_step >= self.max_steps:
            control.should_training_stop = True


def sweep_eval_steps(num_examples: int, batch_size: int, evaluations: int) -> int:
    """Optimizer steps between evaluations so a trial is evaluated `evaluations` times."""
    return max(1, steps_per_epoch(num_examples, batch_size) // evaluations)


def sweep_results(study: optuna.Study) -> list[dict[str, Any]]:
    """One row per finished trial: number (1-based), status, hyperparameters, best dev loss."""
    rows: list[dict[str, Any]] = []
    for trial in study.get_trials(deepcopy=False):
        curve: list[float] = trial.user_attrs.get("dev_combined_curve", [])
        if not curve:
            continue
        rows.append(
            {
                "trial": trial.number + 1,
                "status": trial.state.name.lower(),
                "mlflow_run_id": trial.user_attrs.get("mlflow_run_id"),
                **trial.params,
                "best_dev_combined_loss": min(curve),
                "evaluations": len(curve),
            }
        )
    return rows


def write_sweep_results(
    study: optuna.Study, finished_trial: optuna.trial.FrozenTrial, results_path: str
) -> None:
    """Optuna callback: rewrite the sweep results file after every finished trial."""
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sweep_results(study), indent=4))


def sweep_objective(
    trial: optuna.Trial, config: MultitaskConfig, splits: DatasetDict, model_source: str
) -> float:
    """Train one trial on the sweep budget and return its best combined dev loss.

    The schedule (warmup, decay) is set for the whole mixture, as in a full
    run, and the trial stops after `sweep.train_epochs` mixture epochs. It is
    evaluated `sweep.evals_per_epoch` times per epoch on the dev sets only (no
    test data) and can be pruned at every evaluation. Nothing is saved.

    Raises:
        optuna.TrialPruned: If the pruner stops the trial.
    """
    sweep = config.sweep
    hyperparameters: Hyperparameters = suggest_hyperparameters(trial, sweep.search_space)
    print(f"\nSweep trial {trial.number + 1}/{sweep.n_trials}: {hyperparameters}")

    stats: dict[str, Any] = json.loads(
        (Path(config.multitask.processed_dir) / "stats.json").read_text()
    )
    num_examples: int = int(stats["examples_per_epoch"] * sweep.train_epochs)
    evaluations: int = max(1, round(sweep.evals_per_epoch * sweep.train_epochs))
    batch_size: int = int(hyperparameters["batch_size"])
    eval_steps: int = sweep_eval_steps(num_examples, batch_size, evaluations)
    budget_steps: int = eval_steps * evaluations
    trial_dir = Path(sweep.trials_dir) / f"trial_{trial.number + 1}"
    pruning_callback = OptunaPruningCallback(trial, metric_name="eval_combined_loss")
    pruned = False

    with start_stage_run(
        "finetune",
        f"multitask-sweep-trial-{trial.number + 1}",
        config,
        {"task": "multitask_sweep", "trial_number": str(trial.number + 1)},
        nested=True,
    ) as run:
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        trainer = SFTTrainer(
            model=AutoModelForCausalLM.from_pretrained(model_source),
            args=build_multitask_sft_config(
                config, trial_dir, config.seed, eval_steps, hyperparameters, save_models=False
            ),
            # The whole mixture, so the schedule has the length of a full run
            train_dataset=splits["train"].select_columns(SFT_COLUMNS),
            eval_dataset={name: splits[f"dev_{name}"].select_columns(SFT_COLUMNS) for name in DEV_SETS},
            processing_class=AutoTokenizer.from_pretrained(model_source),
            callbacks=[pruning_callback, StopAfterStepsCallback(budget_steps)],
        )
        try:
            trainer.train()
            log_history: list[dict[str, float]] = trainer.state.log_history
        except optuna.TrialPruned:
            pruned = True
            log_history = pruning_callback.log_history

        curve: list[float] = epoch_metrics(log_history, ["combined"])["combined"]
        trial.set_user_attr("dev_combined_curve", curve)
        mlflow.set_tag("status", "pruned" if pruned else "complete")
        mlflow.log_metric("best_dev_combined_loss", min(curve))

    del trainer
    free_gpu_memory()
    shutil.rmtree(trial_dir, ignore_errors=True)
    if pruned:
        raise optuna.TrialPruned(f"Pruned after {len(curve)} evaluations")
    return min(curve)


def run_multitask_sweep(config: MultitaskConfig) -> tuple[dict[str, Any], list[dict[str, Any]], optuna.Study]:
    """Optuna search (TPE + median pruning) over the multitask fine-tuning hyperparameters.

    Always starts from `config.model.model_id` (the original GPT-2), so the
    pretrained model is later fine-tuned with values it did not influence.
    Logs a `multitask-sweep` MLflow run with one nested run per trial.

    Args:
        config: Multitask configuration (see the `sweep` section).

    Returns:
        The best completed trial, all trial rows and the Optuna study.
    """
    sweep = config.sweep
    splits: DatasetDict = load_multitask(config.multitask.processed_dir)
    model_source: str = resolve_model_source(config.model.model_id)
    configure_tracking(config)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=sweep.pruner_startup_trials, n_warmup_steps=sweep.pruner_warmup_evals
        ),
    )

    with start_stage_run("finetune", "multitask-sweep", config, {"task": "multitask_sweep"}):
        mlflow.log_dict(asdict(config), "config.json")
        study.optimize(
            partial(sweep_objective, config=config, splits=splits, model_source=model_source),
            n_trials=sweep.n_trials,
            callbacks=[partial(write_sweep_results, results_path=sweep.results_path)],
        )
        shutil.rmtree(sweep.trials_dir, ignore_errors=True)

        rows: list[dict[str, Any]] = sweep_results(study)
        completed: list[dict[str, Any]] = [row for row in rows if row["status"] == "complete"]
        if not completed:
            raise ValueError("No sweep trial completed")
        best: dict[str, Any] = min(completed, key=itemgetter("best_dev_combined_loss"))
        mlflow.log_params({f"best_{name}": value for name, value in study.best_trial.params.items()})
        mlflow.log_metric("best_dev_combined_loss", best["best_dev_combined_loss"])
        mlflow.log_artifact(sweep.results_path)

    print("\nBest sweep trial:")
    print(best)
    return best, rows, study


def main() -> None:
    parser = argparse.ArgumentParser(description="Multitask fine-tuning on MedText, PubMedQA and MedMCQA.")
    parser.add_argument("--config", default="configs/gpt2_multitask.yaml")
    parser.add_argument("--model-id", help="Base model (default: model.model_id from the config)")
    parser.add_argument(
        "--sweep", action="store_true", help="Run the Optuna search from the sweep section instead"
    )
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.config)
    if args.sweep:
        run_multitask_sweep(config)
        return

    outcome = run_multitask_seeds(config, args.model_id)
    print(json.dumps(outcome["summary"], indent=2))


if __name__ == "__main__":
    main()
