"""Continued (domain-adaptive) pretraining of a causal LM on the packed PubMed corpus.

Run from the repository root (Hydra overrides pick the base model; GPT-2 by default):

    uv run python -m medllm.training.pretrain model=qwen3-1.7b
    uv run python -m medllm.training.pretrain model=qwen3-1.7b --max-steps 20
    uv run python -m medllm.training.pretrain model=qwen3-1.7b --sweep

A full run writes `results/<model>/pretrain.json` (the DVC stage output).

A full run resumes from the last checkpoint in `pretraining.output_dir` (and
continues the same MLflow run) if it was interrupted. Runs limited with
`--max-steps` or `--max-train-blocks` are smoke tests: they always start
fresh in `<output_dir>-smoke` and are never registered.
"""

import argparse
import math
import shutil
import tempfile
from dataclasses import asdict, replace
from operator import itemgetter
from pathlib import Path
from typing import Any

import mlflow
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from medllm.config import PretrainConfig, load_pretrain_config
from medllm.data.corpus import load_pretrain_corpus
from medllm.modeling import free_gpu_memory, load_causal_lm, resolve_model_source
from medllm.reporting.report import save_stage_result
from medllm.tracking import configure_tracking, log_best_model, start_stage_run

RUN_ID_FILE = "mlflow_run_id.txt"


def collate_blocks(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    """Stack fixed-length token blocks; labels are the inputs themselves.

    The model shifts labels internally. EOS tokens between documents stay
    prediction targets (a padding-based collator would mask them, because
    GPT-2 uses EOS as its padding token).

    Args:
        features: Blocks of equal length with `input_ids`.

    Returns:
        `input_ids` and `labels` tensors of shape (batch, block_size).
    """
    input_ids = torch.tensor(
        [feature["input_ids"] for feature in features], dtype=torch.long
    )
    return {"input_ids": input_ids, "labels": input_ids.clone()}


class PerplexityCallback(TrainerCallback):
    """Log `eval_perplexity` (exp of the evaluation loss) to the active MLflow run."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None or "eval_loss" not in metrics or mlflow.active_run() is None:
            return
        mlflow.log_metric(
            "eval_perplexity", math.exp(metrics["eval_loss"]), step=state.global_step
        )


def build_training_arguments(
    config: PretrainConfig, output_dir: Path, max_steps: int | None
) -> TrainingArguments:
    """Trainer arguments for continued pretraining.

    Args:
        config: Pretraining configuration.
        output_dir: Directory for checkpoints.
        max_steps: Stop after this many optimizer steps (smoke tests); None
            trains for `num_train_epochs`.

    Returns:
        The training arguments (bf16, step-based evaluation and checkpoints,
        metrics reported to MLflow).
    """
    training = config.pretraining
    eval_steps: int = training.eval_steps
    save_steps: int = training.save_steps
    if max_steps is not None:
        eval_steps = min(eval_steps, max_steps)
        save_steps = min(save_steps, max_steps)

    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.lr_scheduler_type,
        # A float in [0, 1) is a fraction of the total training steps
        warmup_steps=training.warmup_ratio,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        num_train_epochs=training.num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=training.per_device_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        per_device_eval_batch_size=training.eval_batch_size,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=training.save_total_limit,
        logging_strategy="steps",
        logging_steps=training.logging_steps,
        bf16=training.bf16,
        gradient_checkpointing=training.gradient_checkpointing,
        torch_compile=training.torch_compile,
        optim=training.optim,
        dataloader_num_workers=training.dataloader_num_workers,
        seed=config.seed,
        report_to="mlflow",
    )


def select_blocks(
    corpus: DatasetDict, max_train_blocks: int | None
) -> tuple[Dataset, Dataset]:
    """Train and validation blocks, optionally keeping only the first training blocks."""
    train_blocks: Dataset = corpus["train"]
    if max_train_blocks is not None:
        train_blocks = train_blocks.select(
            range(min(max_train_blocks, len(train_blocks)))
        )
    return train_blocks, corpus["validation"]


def run_pretraining(
    config: PretrainConfig,
    max_steps: int | None = None,
    max_train_blocks: int | None = None,
    register: bool = True,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Continue pretraining the configured model on the packed corpus.

    Args:
        config: Pretraining configuration.
        max_steps: Optimizer-step limit for a smoke test.
        max_train_blocks: Use only this many training blocks (smoke test).
        register: Register the final model in the Model Registry (full runs only).
        run_name: MLflow run name (defaults to `pretrain` or `pretrain-smoke`).
        extra_tags: Additional MLflow tags (they override the default `run_type`).
        output_dir: Checkpoint directory (defaults to `pretraining.output_dir`,
            with `-smoke` appended for smoke tests). Smoke directories are
            emptied first.

    Returns:
        The MLflow run id, the final model directory, the final validation
        loss and perplexity, and the registered model id (None if not registered).
    """
    training = config.pretraining
    is_smoke: bool = max_steps is not None or max_train_blocks is not None
    if output_dir is None:
        output_dir = training.output_dir + ("-smoke" if is_smoke else "")
    output_dir = Path(output_dir)
    if is_smoke:
        shutil.rmtree(output_dir, ignore_errors=True)
    checkpoint: str | None = (
        get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    )
    run_id_path: Path = output_dir / RUN_ID_FILE

    train_blocks, validation_blocks = select_blocks(
        load_pretrain_corpus(config.corpus.processed_dir), max_train_blocks
    )
    model_source: str = resolve_model_source(config.model.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = load_causal_lm(model_source)

    configure_tracking(config)
    if checkpoint is not None and run_id_path.exists():
        print(f"Resuming from {checkpoint}")
        active_run = mlflow.start_run(run_id=run_id_path.read_text().strip())
    else:
        tags: dict[str, str] = {"run_type": "smoke" if is_smoke else "full", **(extra_tags or {})}
        active_run = start_stage_run(
            "pretrain",
            run_name or ("pretrain-smoke" if is_smoke else "pretrain"),
            config,
            tags,
            nested=mlflow.active_run() is not None,
        )

    with active_run as run:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run.info.run_id)
        if checkpoint is None:
            mlflow.log_dict(asdict(config), "config.json")
            mlflow.log_artifact(
                str(Path(config.corpus.processed_dir) / "stats.json"), "corpus"
            )
            mlflow.log_params(
                {
                    "corpus_dir": config.corpus.processed_dir,
                    "train_blocks": len(train_blocks),
                    "validation_blocks": len(validation_blocks),
                    "tokens_per_step": training.per_device_batch_size
                    * training.gradient_accumulation_steps
                    * config.model.max_seq_len,
                }
            )

        trainer = Trainer(
            model=model,
            args=build_training_arguments(config, output_dir, max_steps),
            train_dataset=train_blocks,
            eval_dataset=validation_blocks,
            data_collator=collate_blocks,
            processing_class=tokenizer,
            callbacks=[PerplexityCallback()],
        )
        trainer.train(resume_from_checkpoint=checkpoint)

        final_metrics: dict[str, float] = trainer.evaluate()
        final_loss: float = final_metrics["eval_loss"]
        final_dir: Path = output_dir / "final"
        trainer.save_model(str(final_dir))
        mlflow.log_metrics(
            {
                "final_eval_loss": final_loss,
                "final_eval_perplexity": math.exp(final_loss),
            }
        )

    model_id: str | None = None
    if register and not is_smoke:
        model_id = log_best_model(
            run.info.run_id,
            final_dir,
            config.mlflow.registered_model_name,
            training.registry_alias,
        )

    return {
        "run_id": run.info.run_id,
        "model_dir": str(final_dir),
        "eval_loss": final_loss,
        "eval_perplexity": math.exp(final_loss),
        "model_id": model_id,
    }


def evaluate_on_corpus(model_id: str, config: PretrainConfig) -> dict[str, float]:
    """Loss and perplexity of a model on the corpus validation blocks.

    Uses the same blocks, batch size and precision as the evaluations during
    pretraining, so base and pretrained models are directly comparable.

    Args:
        model_id: Hub id, local directory or MLflow model URI.
        config: Pretraining configuration.

    Returns:
        `loss` and `perplexity`.
    """
    model = load_causal_lm(model_id)
    validation_blocks: Dataset = load_pretrain_corpus(config.corpus.processed_dir)[
        "validation"
    ]

    with tempfile.TemporaryDirectory() as output_dir:
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=config.pretraining.eval_batch_size,
            bf16=config.pretraining.bf16,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=validation_blocks,
            data_collator=collate_blocks,
        )
        loss: float = trainer.evaluate()["eval_loss"]

    return {"loss": loss, "perplexity": math.exp(loss)}


def with_learning_rate(config: PretrainConfig, learning_rate: float) -> PretrainConfig:
    """Copy of the configuration with another peak learning rate."""
    return replace(config, pretraining=replace(config.pretraining, learning_rate=learning_rate))


def sweep_learning_rates(config: PretrainConfig) -> list[dict[str, Any]]:
    """Short pretraining runs, one per peak learning rate in `config.lr_sweep`.

    Logs an `lr-sweep` MLflow run with one nested run per learning rate. Each
    run decays its schedule over `lr_sweep.max_steps`. Final models are kept
    in `<output_dir>-lr-sweep/lr-<rate>/final` (so they can also be evaluated
    on the fine-tuning data); intermediate checkpoints are deleted.

    Args:
        config: Pretraining configuration.

    Returns:
        One result per learning rate (see `run_pretraining`), plus `learning_rate`.
    """
    sweep = config.lr_sweep
    sweep_dir = Path(config.pretraining.output_dir + "-lr-sweep")
    results: list[dict[str, Any]] = []

    configure_tracking(config)
    with start_stage_run("pretrain", "lr-sweep", config, {"run_type": "lr_sweep"}):
        mlflow.log_params(
            {"sweep_learning_rates": str(sweep.learning_rates), "sweep_max_steps": sweep.max_steps}
        )
        for learning_rate in sweep.learning_rates:
            run_dir: Path = sweep_dir / f"lr-{learning_rate:g}"
            result = run_pretraining(
                with_learning_rate(config, learning_rate),
                max_steps=sweep.max_steps,
                register=False,
                run_name=f"lr-sweep-{learning_rate:g}",
                extra_tags={"run_type": "lr_sweep"},
                output_dir=run_dir,
            )
            for checkpoint_dir in run_dir.glob("checkpoint-*"):
                shutil.rmtree(checkpoint_dir)
            results.append({"learning_rate": learning_rate, **result})
            free_gpu_memory()

        best = min(results, key=itemgetter("eval_loss"))
        mlflow.log_params({"best_learning_rate": best["learning_rate"]})
        mlflow.log_metrics({"best_eval_perplexity": best["eval_perplexity"]})
        mlflow.log_dict({"results": results}, "lr_sweep_results.json")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continued pretraining on the PubMed corpus."
    )
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Smoke test: stop after N optimizer steps"
    )
    parser.add_argument(
        "--max-train-blocks",
        type=int,
        help="Smoke test: use only the first N training blocks",
    )
    parser.add_argument(
        "--no-register", action="store_true", help="Do not register the final model"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run the learning-rate sweep from lr_sweep"
    )
    args = parser.parse_args()

    config: PretrainConfig = load_pretrain_config(args.overrides)

    if args.sweep:
        for result in sweep_learning_rates(config):
            print(result)
        return

    result = run_pretraining(
        config,
        max_steps=args.max_steps,
        max_train_blocks=args.max_train_blocks,
        register=not args.no_register,
    )
    print(result)
    if result["model_id"] is not None:
        model_uri = f"models:/{config.mlflow.registered_model_name}@{config.pretraining.registry_alias}"
        save_stage_result(config.results_dir, "pretrain", {**result, "model_uri": model_uri})


if __name__ == "__main__":
    main()
