"""Optuna search (TPE + median pruning) over the multitask fine-tuning hyperparameters.

    uv run python -m medllm.training.sweep model=qwen3-1.7b

Trials train on the full mixture schedule but stop after `sweep.train_epochs`
mixture epochs, are scored on the validation sets only, and can be pruned at
every evaluation. The chosen values go into the model file
(configs/model/<model>.yaml), so the pipeline runs stay reproducible from git.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any

import mlflow
import optuna
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

from medllm.config import (
    FloatRange,
    Hyperparameters,
    MultitaskConfig,
    SearchSpace,
    load_multitask_config,
)
from medllm.data.finetune_data import load_finetune_data
from medllm.data.mixture import build_mixture
from medllm.data.preprocessing import SFT_COLUMNS
from medllm.modeling import (
    build_peft_config,
    free_gpu_memory,
    load_causal_lm,
    resolve_model_source,
    weights_dtype,
)
from medllm.tracking import (
    SectionedMLflowCallback,
    configure_tracking,
    metric_section_name,
    start_stage_run,
)
from medllm.training.finetune import (
    VALIDATION_SETS,
    build_sft_config,
    epoch_metrics,
    steps_per_epoch,
)


class OptunaPruningCallback(TrainerCallback):
    """Report a validation metric to Optuna after every evaluation and stop the
    run when the pruner judges the trial worse than earlier trials at that point.

    With several eval datasets the trainer calls `on_evaluate` once per
    dataset; only calls that contain `metric_name` are reported.
    """

    def __init__(self, trial: optuna.Trial, metric_name: str = "eval_loss") -> None:
        self.trial = trial
        self.metric_name = metric_name
        self.reports = 0
        self.log_history: list[dict[str, float]] = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None or self.metric_name not in metrics:
            return

        # Kept so a pruned trial's losses can still be recorded
        self.log_history = list(state.log_history)
        # Steps start at 0: MedianPruner skips steps below n_warmup_steps, so
        # n_warmup_steps=1 protects exactly the first evaluation
        self.trial.report(metrics[self.metric_name], step=self.reports)
        self.reports += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at evaluation {self.reports}")


def suggest_hyperparameters(
    trial: optuna.Trial, search_space: SearchSpace
) -> Hyperparameters:
    """Sample one value per hyperparameter from the search space.

    Args:
        trial: Current Optuna trial.
        search_space: Float ranges and categorical choices.

    Returns:
        The sampled hyperparameters.
    """
    hyperparameters: Hyperparameters = {}

    for name, space in search_space.items():
        if isinstance(space, FloatRange):
            hyperparameters[name] = trial.suggest_float(
                name, space.low, space.high, log=space.log
            )
        else:
            hyperparameters[name] = trial.suggest_categorical(name, space)

    return hyperparameters


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
    """One row per finished trial: number (1-based), status, hyperparameters, best validation loss."""
    rows: list[dict[str, Any]] = []
    for trial in study.get_trials(deepcopy=False):
        curve: list[float] = trial.user_attrs.get("validation_combined_curve", [])
        if not curve:
            continue
        rows.append(
            {
                "trial": trial.number + 1,
                "status": trial.state.name.lower(),
                "mlflow_run_id": trial.user_attrs.get("mlflow_run_id"),
                **trial.params,
                "best_validation_combined_loss": min(curve),
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
    """Train one trial on the sweep budget and return its best combined validation loss.

    The schedule (warmup, decay) is set for the whole mixture, as in a full
    run, and the trial stops after `sweep.train_epochs` mixture epochs. It is
    evaluated `sweep.evals_per_epoch` times per epoch on the validation sets only (no
    test data) and can be pruned at every evaluation. Nothing is saved.

    Raises:
        optuna.TrialPruned: If the pruner stops the trial.
    """
    sweep = config.sweep
    hyperparameters: Hyperparameters = suggest_hyperparameters(trial, sweep.search_space)
    print(f"\nSweep trial {trial.number + 1}/{sweep.n_trials}: {hyperparameters}")

    train_set: Dataset = build_mixture(splits, config.mixture, config.seed)
    num_examples: int = int(len(train_set) // config.mixture.epochs * sweep.train_epochs)
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
            model=load_causal_lm(model_source, weights_dtype(config.finetune.method)),
            args=build_sft_config(
                config, trial_dir, config.seed, eval_steps, hyperparameters, save_models=False
            ),
            # The whole mixture, so the schedule has the length of a full run
            train_dataset=train_set.select_columns(SFT_COLUMNS),
            eval_dataset={name: splits[f"validation_{name}"].select_columns(SFT_COLUMNS) for name in VALIDATION_SETS},
            processing_class=AutoTokenizer.from_pretrained(model_source),
            callbacks=[SectionedMLflowCallback(), pruning_callback, StopAfterStepsCallback(budget_steps)],
            peft_config=build_peft_config(config),
        )
        try:
            trainer.train()
            log_history: list[dict[str, float]] = trainer.state.log_history
        except optuna.TrialPruned:
            pruned = True
            log_history = pruning_callback.log_history

        curve: list[float] = epoch_metrics(log_history, ["combined"])["combined"]
        trial.set_user_attr("validation_combined_curve", curve)
        mlflow.set_tag("status", "pruned" if pruned else "complete")
        mlflow.log_metric(metric_section_name("best_validation_combined_loss"), min(curve))

    del trainer
    free_gpu_memory()
    shutil.rmtree(trial_dir, ignore_errors=True)
    if pruned:
        raise optuna.TrialPruned(f"Pruned after {len(curve)} evaluations")
    return min(curve)


def run_multitask_sweep(config: MultitaskConfig) -> tuple[dict[str, Any], list[dict[str, Any]], optuna.Study]:
    """Optuna search (TPE + median pruning) over the multitask fine-tuning hyperparameters.

    Always starts from `config.model.model_id` (the Hub model), so the
    pretrained model is later fine-tuned with values it did not influence.
    Logs a `multitask-sweep` MLflow run with one nested run per trial.

    Args:
        config: Multitask configuration (see the `sweep` section).

    Returns:
        The best completed trial, all trial rows and the Optuna study.
    """
    sweep = config.sweep
    splits: DatasetDict = load_finetune_data(config.finetune_data.processed_dir)
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
        best: dict[str, Any] = min(completed, key=itemgetter("best_validation_combined_loss"))
        mlflow.log_params({f"best_{name}": value for name, value in study.best_trial.params.items()})
        mlflow.log_metric(metric_section_name("best_validation_combined_loss"), best["best_validation_combined_loss"])
        mlflow.log_artifact(sweep.results_path)

    print("\nBest sweep trial:")
    print(best)
    return best, rows, study


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search over the multitask fine-tuning hyperparameters.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    args = parser.parse_args()

    run_multitask_sweep(load_multitask_config(args.overrides))


if __name__ == "__main__":
    main()
