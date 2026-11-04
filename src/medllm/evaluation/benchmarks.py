"""Standard medical benchmarks with the lm-evaluation-harness, comparable with published results.

    uv run python -m medllm.evaluation.benchmarks --runs lora_original model=qwen3-1.7b

Scores the base model (zero-shot) and every seed of the given fine-tuning runs
on `benchmarks.tasks` (MedQA, MedMCQA and PubMedQA as multiple choice with the
options shown, and the medical subjects of MMLU). Unlike `medllm.evaluation.tasks`,
which asks the questions open-ended, these are the standard formats, so the
numbers can be compared with other models. Every model is asked in the chat
format it is trained on (system prompt, thinking off).

Writes `results/<model>/benchmarks.json` and logs `benchmark_<task>_acc` to the
MLflow run of each model.
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import mlflow
import torch
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from mlflow.entities import Metric

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.reporting.report import ZERO_SHOT, load_stage_result, save_stage_result
from medllm.tracking import configure_tracking
from medllm.training.finetune import summarize_seeds

METRIC_PREFIX = "benchmark"
# Tasks of this project (pubmedqa_test), next to the harness's own
TASKS_DIR = Path(__file__).resolve().parent / "lm_eval_tasks"


def harness_model_args(model_path: str) -> dict[str, Any]:
    """Arguments of the harness's Hugging Face model: bf16 weights, thinking off (as in training)."""
    return {"pretrained": model_path, "dtype": "bfloat16", "enable_thinking": False}


def benchmark_metrics(results: dict[str, Any], tasks: list[str]) -> dict[str, float]:
    """Accuracy and its standard error per task, from `simple_evaluate`'s output.

    Args:
        results: Output of `lm_eval.simple_evaluate`.
        tasks: Tasks to keep.

    Returns:
        `benchmark_<task>_acc` and `benchmark_<task>_acc_stderr` for every task.

    Raises:
        KeyError: If a task has no results.
    """
    metrics: dict[str, float] = {}
    for task in tasks:
        if task not in results["results"]:
            raise KeyError(f"No benchmark results for {task}")
        task_results: dict[str, Any] = results["results"][task]
        metrics[f"{METRIC_PREFIX}_{task}_acc"] = float(task_results["acc,none"])
        metrics[f"{METRIC_PREFIX}_{task}_acc_stderr"] = float(task_results["acc_stderr,none"])
    return metrics


def benchmark_model(model_path: str, config: MultitaskConfig) -> dict[str, float]:
    """Run every benchmark task on one model; the model is freed from the GPU afterwards."""
    print(f"Benchmarking {model_path}")
    results: dict[str, Any] = simple_evaluate(
        model="hf",
        model_args=harness_model_args(model_path),
        tasks=list(config.benchmarks.tasks),
        num_fewshot=config.benchmarks.num_fewshot,
        batch_size=config.benchmarks.batch_size,
        apply_chat_template=True,
        system_instruction=config.multitask.system_prompt,
        task_manager=TaskManager(include_path=str(TASKS_DIR)),
    )
    metrics: dict[str, float] = benchmark_metrics(results, list(config.benchmarks.tasks))
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def run_benchmarks(config: MultitaskConfig, runs: list[str]) -> dict[str, Any]:
    """Benchmark the base model and every seed of the given fine-tuning runs.

    Args:
        config: Multitask configuration (see the `benchmarks` section).
        runs: Fine-tuning results to benchmark, by the suffix of their file
            (`lora_original` for `results/<model>/finetune_lora_original.json`).

    Returns:
        The tasks, and per run (`zero-shot` first) the metrics of every seed
        and their mean and standard deviation.
    """
    configure_tracking(config)
    client = mlflow.MlflowClient()
    baseline: dict[str, Any] = load_stage_result(config.results_dir, "baseline")
    zero_shot: dict[str, float] = benchmark_model(config.model.model_id, config)
    client.log_batch(baseline["run_id"], metrics=[Metric(key, value, 0, 0) for key, value in zero_shot.items()])
    outcome: dict[str, Any] = {
        "tasks": list(config.benchmarks.tasks),
        "num_fewshot": config.benchmarks.num_fewshot,
        "runs": {
            ZERO_SHOT: {
                "seeds": [{"run_id": baseline["run_id"], "metrics": zero_shot}],
                "summary": summarize_seeds([zero_shot]),
            }
        },
    }

    for run in runs:
        finetune: dict[str, Any] = load_stage_result(config.results_dir, f"finetune_{run}")
        seeds: list[dict[str, Any]] = []
        for seed in finetune["seeds"]:
            metrics: dict[str, float] = benchmark_model(seed["model_dir"], config)
            client.log_batch(seed["run_id"], metrics=[Metric(key, value, 0, 0) for key, value in metrics.items()])
            seeds.append({"run_id": seed["run_id"], "metrics": metrics})
        summary: dict[str, dict[str, float]] = summarize_seeds([seed["metrics"] for seed in seeds])
        client.log_batch(
            finetune["parent_run_id"],
            metrics=[Metric(f"{key}_mean", value["mean"], 0, 0) for key, value in summary.items()],
        )
        outcome["runs"][run] = {"seeds": seeds, "summary": summary}
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard medical benchmarks (lm-evaluation-harness).")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    parser.add_argument("--runs", nargs="+", required=True, help="Fine-tuning runs, e.g. lora_original")
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    outcome: dict[str, Any] = run_benchmarks(config, args.runs)
    save_stage_result(config.results_dir, "benchmarks", outcome)
    print(json.dumps({run: value["summary"] for run, value in outcome["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
