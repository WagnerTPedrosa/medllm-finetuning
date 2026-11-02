"""Fine-tuning with SFTTrainer and hyperparameter search (grid and Optuna)."""

import gc
import json
import shutil
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, TypedDict

import mlflow
import optuna
import torch
from datasets import Dataset
from sklearn.model_selection import ParameterGrid
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from src.config import Config, FloatRange, Hyperparameters, SearchSpace
from src.modeling import resolve_model_source
from src.tracking import (
    configure_tracking,
    log_best_model,
    log_search_inputs,
    log_trial_metrics,
    log_trial_parameters,
    start_stage_run,
)

TrialResult = dict[str, Any]


class TrialMetrics(TypedDict):
    best_val_loss: float
    best_epoch: int
    train_losses: list[float]
    val_losses: list[float]


def losses_by_epoch(
    log_history: list[dict[str, float]],
) -> tuple[list[float], list[float]]:
    """Split Trainer logs into per-epoch train and validation losses.

    Args:
        log_history: `trainer.state.log_history`, logged once per epoch.

    Returns:
        The train losses and the validation losses, in epoch order.
    """
    train_losses: list[float] = [log["loss"] for log in log_history if "loss" in log]
    val_losses: list[float] = [
        log["eval_loss"] for log in log_history if "eval_loss" in log
    ]
    return train_losses, val_losses


def summarize_losses(log_history: list[dict[str, float]]) -> TrialMetrics:
    """Summarize a run's per-epoch losses and find its best epoch.

    Args:
        log_history: `trainer.state.log_history`.

    Returns:
        The lowest validation loss, its epoch (1-based) and both loss curves.

    Raises:
        ValueError: If the logs contain no validation loss.
    """
    train_losses, val_losses = losses_by_epoch(log_history)
    if not val_losses:
        raise ValueError("log_history has no eval_loss entries")

    best_val_loss: float = min(val_losses)
    return {
        "best_val_loss": best_val_loss,
        "best_epoch": val_losses.index(best_val_loss) + 1,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def free_gpu_memory() -> None:
    """Release cached GPU memory between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_sft_config(
    hyperparameters: Hyperparameters, config: Config, output_dir: Path
) -> SFTConfig:
    """Build the SFTTrainer arguments for one trial.

    Args:
        hyperparameters: Needs `learning_rate` and `batch_size`; may override
            `weight_decay`, `lr_scheduler_type` and `warmup_ratio`.
        config: Experiment configuration (provides the defaults).
        output_dir: Directory for this trial's checkpoints.

    Returns:
        The SFTTrainer configuration.
    """
    training = config.training
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training.max_epochs,
        learning_rate=hyperparameters["learning_rate"],
        per_device_train_batch_size=hyperparameters["batch_size"],
        per_device_eval_batch_size=training.eval_batch_size,
        weight_decay=hyperparameters.get("weight_decay", training.weight_decay),
        max_grad_norm=training.max_grad_norm,
        lr_scheduler_type=hyperparameters.get(
            "lr_scheduler_type", training.lr_scheduler_type
        ),
        # A float in [0, 1) is a fraction of the total training steps
        warmup_steps=hyperparameters.get("warmup_ratio", training.warmup_ratio),
        # Loss only on the answer tokens (prompt tokens get label -100)
        completion_only_loss=True,
        max_length=config.model.max_seq_len,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )


def train_trial(
    train_dataset: Dataset,
    val_dataset: Dataset,
    hyperparameters: Hyperparameters,
    config: Config,
    output_dir: Path,
    extra_callbacks: list[TrainerCallback] | None = None,
    model_source: str | None = None,
) -> tuple[SFTTrainer, TrialMetrics]:
    """Fine-tune a fresh pretrained model and keep its best epoch.

    Training stops once validation loss has not improved for
    `config.training.early_stopping_patience` epochs, then the best epoch's
    weights are reloaded.

    Args:
        train_dataset: Prompt-completion training split.
        val_dataset: Prompt-completion validation split.
        hyperparameters: See `build_sft_config`.
        config: Experiment configuration.
        output_dir: Directory for this trial's checkpoints.
        extra_callbacks: Callbacks added after early stopping.
        model_source: Hub id or local directory to start from; defaults to
            `config.model.model_id` (see `src.modeling.resolve_model_source`).

    Returns:
        The trainer (holding the best epoch's model) and the trial's losses.
    """
    source: str = model_source or resolve_model_source(config.model.model_id)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source)
    model.config.use_cache = False

    callbacks: list[TrainerCallback] = [
        EarlyStoppingCallback(config.training.early_stopping_patience),
        *(extra_callbacks or []),
    ]
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(hyperparameters, config, output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()

    return trainer, summarize_losses(trainer.state.log_history)


def find_best_result(results: list[TrialResult]) -> TrialResult:
    """Return the completed trial with the lowest validation loss.

    Args:
        results: Trial results; entries without a `status` count as completed.

    Returns:
        The best completed trial.

    Raises:
        ValueError: If no trial completed.
    """
    completed: list[TrialResult] = [
        result for result in results if result.get("status", "complete") == "complete"
    ]
    if not completed:
        raise ValueError("No completed trials")

    return min(completed, key=itemgetter("best_val_loss"))


def write_results(results: list[TrialResult], results_path: str | Path) -> None:
    """Write trial results to a JSON file, creating its directory if needed."""
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=4))


def run_grid_search(
    train_dataset: Dataset, val_dataset: Dataset, config: Config
) -> tuple[TrialResult, list[TrialResult]]:
    """Train one model per combination in `config.grid_search.param_grid`.

    Logs a `grid-search` MLflow run with one nested run per trial. Results are
    written after every trial. The best model is saved to
    `config.grid_search.best_checkpoint_dir`, stored in its trial's run and
    registered as `models:/<registered_model_name>@grid-best`.

    Args:
        train_dataset: Prompt-completion training split.
        val_dataset: Prompt-completion validation split.
        config: Experiment configuration.

    Returns:
        The best trial and all trial results.
    """
    search = config.grid_search
    combinations: list[Hyperparameters] = list(ParameterGrid(search.param_grid))
    results: list[TrialResult] = []
    best_val_loss = float("inf")
    model_source: str = resolve_model_source(config.model.model_id)
    configure_tracking(config)

    with start_stage_run("finetune", "grid-search", config, {"search_kind": "grid"}):
        log_search_inputs(train_dataset, val_dataset, config)

        for trial_number, hyperparameters in enumerate(combinations, start=1):
            print(f"\nTrial {trial_number}/{len(combinations)}")
            print(f"Parameters: {hyperparameters}")

            trial_dir = Path(search.trials_dir) / f"trial_{trial_number}"
            with start_stage_run(
                "finetune", f"grid-trial-{trial_number}", config, nested=True
            ) as run:
                log_trial_parameters("grid", trial_number, hyperparameters, config)
                trainer, metrics = train_trial(
                    train_dataset,
                    val_dataset,
                    hyperparameters,
                    config,
                    trial_dir,
                    model_source=model_source,
                )
                log_trial_metrics(
                    trainer.state.log_history,
                    metrics["best_val_loss"],
                    metrics["best_epoch"],
                )

            if metrics["best_val_loss"] < best_val_loss:
                best_val_loss = metrics["best_val_loss"]
                trainer.save_model(search.best_checkpoint_dir)

            results.append(
                {
                    "trial": trial_number,
                    "mlflow_run_id": run.info.run_id,
                    **hyperparameters,
                    **metrics,
                }
            )
            write_results(results, search.results_path)

            del trainer
            free_gpu_memory()
            shutil.rmtree(trial_dir, ignore_errors=True)

        shutil.rmtree(search.trials_dir, ignore_errors=True)
        best: TrialResult = find_best_result(results)
        log_search_summary(best, search.results_path)
        best["mlflow_model_id"] = log_best_model(
            best["mlflow_run_id"],
            search.best_checkpoint_dir,
            config.mlflow.registered_model_name,
            alias="grid-best",
        )

    write_results(results, search.results_path)
    print("\nBest hyperparameters:")
    print(best)
    return best, results


def log_search_summary(best: TrialResult, results_path: str | Path) -> None:
    """Record the best trial on the active search run, plus the results file."""
    mlflow.log_metrics(
        {"best_val_loss": best["best_val_loss"], "best_epoch": best["best_epoch"]}
    )
    mlflow.log_params(
        {
            f"best_{name}": value
            for name, value in best.items()
            if name in {"learning_rate", "batch_size", "weight_decay",
                        "lr_scheduler_type", "warmup_ratio"}
        }
    )
    mlflow.log_artifact(str(results_path))


class OptunaPruningCallback(TrainerCallback):
    """Report each epoch's validation loss to Optuna and stop the run when the
    pruner judges the trial worse than earlier trials at that epoch."""

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial
        self.log_history: list[dict[str, float]] = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        # Kept so a pruned trial's losses can still be recorded
        self.log_history = list(state.log_history)
        epoch: int = len(losses_by_epoch(self.log_history)[1])

        self.trial.report(metrics["eval_loss"], step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}")


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


def best_completed_value(study: optuna.Study) -> float:
    """Return the lowest objective among completed trials (inf if none)."""
    completed = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    return min((trial.value for trial in completed), default=float("inf"))


def objective(
    trial: optuna.Trial,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Config,
    model_source: str,
) -> float:
    """Optuna objective: train one model and return its best validation loss.

    Logs a nested MLflow run, stores the trial's losses in
    `trial.user_attrs["metrics"]` (also for pruned trials) and saves the model
    to `config.optuna.best_checkpoint_dir` when it beats every completed trial.

    Args:
        trial: Current Optuna trial.
        train_dataset: Prompt-completion training split.
        val_dataset: Prompt-completion validation split.
        config: Experiment configuration.
        model_source: Hub id or local directory to start from.

    Returns:
        The trial's best validation loss.

    Raises:
        optuna.TrialPruned: If the pruner stops the trial.
    """
    search = config.optuna
    hyperparameters: Hyperparameters = suggest_hyperparameters(
        trial, search.search_space
    )
    print(f"\nTrial {trial.number + 1}/{search.n_trials}")
    print(f"Parameters: {hyperparameters}")

    trial_dir = Path(search.trials_dir) / f"trial_{trial.number + 1}"
    pruning_callback = OptunaPruningCallback(trial)

    with start_stage_run(
        "finetune", f"optuna-trial-{trial.number + 1}", config, nested=True
    ) as run:
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        log_trial_parameters("optuna", trial.number + 1, hyperparameters, config)
        try:
            trainer, metrics = train_trial(
                train_dataset,
                val_dataset,
                hyperparameters,
                config,
                trial_dir,
                [pruning_callback],
                model_source=model_source,
            )
        except optuna.TrialPruned:
            metrics = summarize_losses(pruning_callback.log_history)
            trial.set_user_attr("metrics", metrics)
            mlflow.set_tag("status", "pruned")
            log_trial_metrics(
                pruning_callback.log_history,
                metrics["best_val_loss"],
                metrics["best_epoch"],
            )
            free_gpu_memory()
            shutil.rmtree(trial_dir, ignore_errors=True)
            raise

        trial.set_user_attr("metrics", metrics)
        mlflow.set_tag("status", "complete")
        log_trial_metrics(
            trainer.state.log_history, metrics["best_val_loss"], metrics["best_epoch"]
        )

    if metrics["best_val_loss"] < best_completed_value(trial.study):
        trainer.save_model(search.best_checkpoint_dir)

    del trainer
    free_gpu_memory()
    shutil.rmtree(trial_dir, ignore_errors=True)
    return metrics["best_val_loss"]


def study_results(study: optuna.Study) -> list[TrialResult]:
    """Flatten finished trials into the same result format as the grid search.

    Args:
        study: Optuna study.

    Returns:
        One entry per trial with recorded metrics: number (1-based), status
        (`complete` or `pruned`), hyperparameters and losses.
    """
    results: list[TrialResult] = []

    for trial in study.get_trials(deepcopy=False):
        if "metrics" not in trial.user_attrs:
            continue
        results.append(
            {
                "trial": trial.number + 1,
                "status": trial.state.name.lower(),
                "mlflow_run_id": trial.user_attrs.get("mlflow_run_id"),
                **trial.params,
                **trial.user_attrs["metrics"],
            }
        )

    return results


def write_study_results(
    study: optuna.Study, finished_trial: optuna.trial.FrozenTrial, results_path: str
) -> None:
    """Optuna callback: rewrite the results file after every finished trial."""
    write_results(study_results(study), results_path)


def run_optuna_search(
    train_dataset: Dataset, val_dataset: Dataset, config: Config
) -> tuple[TrialResult, list[TrialResult], optuna.Study]:
    """Bayesian hyperparameter search (TPE sampler) with median pruning.

    Each trial trains with early stopping and is pruned when its validation
    loss after an epoch is worse than the median of earlier trials at that
    epoch. Logs an `optuna-search` MLflow run with one nested run per trial.
    The best completed model is saved to `config.optuna.best_checkpoint_dir`,
    stored in its trial's run and registered as
    `models:/<registered_model_name>@optuna-best`.

    Args:
        train_dataset: Prompt-completion training split.
        val_dataset: Prompt-completion validation split.
        config: Experiment configuration.

    Returns:
        The best completed trial, all trial results and the Optuna study.
    """
    search = config.optuna
    model_source: str = resolve_model_source(config.model.model_id)
    configure_tracking(config)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=search.pruner_startup_trials,
            n_warmup_steps=search.pruner_warmup_epochs,
        ),
    )

    with start_stage_run("finetune", "optuna-search", config, {"search_kind": "optuna"}):
        log_search_inputs(train_dataset, val_dataset, config)
        study.optimize(
            partial(
                objective,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=config,
                model_source=model_source,
            ),
            n_trials=search.n_trials,
            callbacks=[partial(write_study_results, results_path=search.results_path)],
        )
        shutil.rmtree(search.trials_dir, ignore_errors=True)

        results: list[TrialResult] = study_results(study)
        best: TrialResult = find_best_result(results)
        log_search_summary(best, search.results_path)
        best["mlflow_model_id"] = log_best_model(
            best["mlflow_run_id"],
            search.best_checkpoint_dir,
            config.mlflow.registered_model_name,
            alias="optuna-best",
        )

    write_results(results, search.results_path)
    print("\nBest hyperparameters:")
    print(best)
    return best, results, study
