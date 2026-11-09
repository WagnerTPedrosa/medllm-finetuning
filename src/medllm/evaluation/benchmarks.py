"""Standard medical benchmarks with the lm-evaluation-harness, comparable with published results.

    uv run python -m medllm.evaluation.benchmarks model=qwen3-1.7b --runs lora_original
    uv run python -m medllm.evaluation.benchmarks model=qwen3-1.7b --runs lora_original lora_dapt --pretrained

Scores the base model (zero-shot), the PubMed-pretrained model before
fine-tuning (`--pretrained`) and every seed of the given fine-tuning runs on
`benchmarks.tasks` (MedQA, MedMCQA and PubMedQA as multiple choice with the
options shown, and the medical subjects of MMLU) and on
`benchmarks.general_tasks` (MMLU, ARC, HellaSwag) and
`benchmarks.perplexity_tasks` (WikiText word perplexity), which show whether
the medical training costs general knowledge. Unlike `medllm.evaluation.tasks`,
which asks the questions open-ended, these are the standard formats, so the
numbers can be compared with other models. Every model is asked in the chat
format it is trained on (system prompt, thinking off).

Writes `results/<model>/benchmarks.json` and logs `benchmark_<task>_acc` to the
MLflow run of each model (under `benchmark/`).
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

from medllm.config import BenchmarksConfig, MultitaskConfig, load_multitask_config
from medllm.reporting.report import (
    ZERO_SHOT,
    ZERO_SHOT_DAPT,
    load_stage_result,
    save_stage_result,
)
from medllm.tracking import configure_tracking, metric_section_name
from medllm.training.finetune import summarize_seeds

METRIC_PREFIX = "benchmark"
# Tasks of this project (pubmedqa_test), next to the harness's own
TASKS_DIR = Path(__file__).resolve().parent / "lm_eval_tasks"


def harness_model_args(model_path: str, max_length: int) -> dict[str, Any]:
    """Arguments of the harness's Hugging Face model: bf16 weights, thinking off and the context of training.

    Args:
        model_path: Hub id or local directory of the model.
        max_length: Longest input in tokens (the models are trained on this
            context; perplexity tasks are scored in windows of this length).

    Returns:
        The `model_args` of `lm_eval.simple_evaluate`.
    """
    return {"pretrained": model_path, "dtype": "bfloat16", "enable_thinking": False, "max_length": max_length}


def benchmark_tasks(benchmarks: BenchmarksConfig) -> list[str]:
    """Every task to run: the medical ones, the general ones, then the perplexity ones.

    A subtask whose group is also run (`mmlu_anatomy` next to `mmlu`) is left
    out: the harness refuses to run a task twice, and the group's results
    already hold the subtask's score.

    Args:
        benchmarks: Benchmark section of the configuration.

    Returns:
        Task names to pass to `lm_eval.simple_evaluate`, in order.
    """
    all_tasks: list[str] = list(benchmarks.tasks) + list(benchmarks.general_tasks) + list(benchmarks.perplexity_tasks)
    # The harness names a group's subtasks `<group>_<subtask>`
    group_prefixes: tuple[str, ...] = tuple(f"{task}_" for task in all_tasks)
    return [task for task in all_tasks if not task.startswith(group_prefixes)]


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


def perplexity_metrics(results: dict[str, Any], tasks: list[str]) -> dict[str, float]:
    """Word perplexity per language-modelling task, from `simple_evaluate`'s output.

    Args:
        results: Output of `lm_eval.simple_evaluate`.
        tasks: Perplexity tasks to keep.

    Returns:
        `benchmark_<task>_word_perplexity` for every task.

    Raises:
        KeyError: If a task has no results.
    """
    metrics: dict[str, float] = {}
    for task in tasks:
        if task not in results["results"]:
            raise KeyError(f"No benchmark results for {task}")
        metrics[f"{METRIC_PREFIX}_{task}_word_perplexity"] = float(results["results"][task]["word_perplexity,none"])
    return metrics


def benchmark_model(model_path: str, config: MultitaskConfig) -> dict[str, float]:
    """Run every benchmark task on one model; the model is freed from the GPU afterwards."""
    print(f"Benchmarking {model_path}")
    results: dict[str, Any] = simple_evaluate(
        model="hf",
        model_args=harness_model_args(model_path, config.model.max_seq_len),
        tasks=benchmark_tasks(config.benchmarks),
        num_fewshot=config.benchmarks.num_fewshot,
        batch_size=config.benchmarks.batch_size,
        apply_chat_template=True,
        system_instruction=config.finetune_data.system_prompt,
        task_manager=TaskManager(include_path=str(TASKS_DIR)),
    )
    accuracy_tasks: list[str] = list(config.benchmarks.tasks) + list(config.benchmarks.general_tasks)
    metrics: dict[str, float] = {
        **benchmark_metrics(results, accuracy_tasks),
        **perplexity_metrics(results, list(config.benchmarks.perplexity_tasks)),
    }
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def benchmark_and_log(client: mlflow.MlflowClient, model_path: str, run_id: str, config: MultitaskConfig) -> dict[str, float]:
    """Benchmark one model and log its metrics to its MLflow run."""
    metrics: dict[str, float] = benchmark_model(model_path, config)
    client.log_batch(run_id, metrics=[Metric(metric_section_name(key), value, 0, 0) for key, value in metrics.items()])
    return metrics


def single_model_result(run_id: str, metrics: dict[str, float]) -> dict[str, Any]:
    """Benchmark result of a model that is not fine-tuned (one seed)."""
    return {"seeds": [{"run_id": run_id, "metrics": metrics}], "summary": summarize_seeds([metrics])}


def run_benchmarks(config: MultitaskConfig, runs: list[str], pretrained: bool = False) -> dict[str, Any]:
    """Benchmark the base model, optionally the pretrained one, and every seed of the given fine-tuning runs.

    Args:
        config: Multitask configuration (see the `benchmarks` section).
        runs: Fine-tuning results to benchmark, by the suffix of their file
            (`lora_original` for `results/<model>/finetune_lora_original.json`).
        pretrained: Also benchmark the PubMed-pretrained model before
            fine-tuning (`results/<model>/pretrain.json`), as `zero-shot_dapt`.

    Returns:
        The medical and general tasks, and per run (`zero-shot` first) the
        metrics of every seed and their mean and standard deviation.
    """
    configure_tracking(config)
    client = mlflow.MlflowClient()
    baseline: dict[str, Any] = load_stage_result(config.results_dir, "baseline")
    zero_shot: dict[str, float] = benchmark_and_log(client, config.model.model_id, baseline["run_id"], config)
    outcome: dict[str, Any] = {
        "tasks": list(config.benchmarks.tasks),
        "general_tasks": list(config.benchmarks.general_tasks),
        "perplexity_tasks": list(config.benchmarks.perplexity_tasks),
        "num_fewshot": config.benchmarks.num_fewshot,
        "runs": {ZERO_SHOT: single_model_result(baseline["run_id"], zero_shot)},
    }

    if pretrained:
        pretrain: dict[str, Any] = load_stage_result(config.results_dir, "pretrain")
        metrics: dict[str, float] = benchmark_and_log(client, pretrain["model_dir"], pretrain["run_id"], config)
        outcome["runs"][ZERO_SHOT_DAPT] = single_model_result(pretrain["run_id"], metrics)

    for run in runs:
        finetune: dict[str, Any] = load_stage_result(config.results_dir, f"finetune_{run}")
        seeds: list[dict[str, Any]] = []
        for seed in finetune["seeds"]:
            metrics = benchmark_and_log(client, seed["model_dir"], seed["run_id"], config)
            seeds.append({"run_id": seed["run_id"], "metrics": metrics})
        summary: dict[str, dict[str, float]] = summarize_seeds([seed["metrics"] for seed in seeds])
        client.log_batch(
            finetune["parent_run_id"],
            metrics=[Metric(metric_section_name(f"{key}_mean"), value["mean"], 0, 0) for key, value in summary.items()],
        )
        outcome["runs"][run] = {"seeds": seeds, "summary": summary}
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard medical benchmarks (lm-evaluation-harness).")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    parser.add_argument("--runs", nargs="+", required=True, help="Fine-tuning runs, e.g. lora_original")
    parser.add_argument(
        "--pretrained", action="store_true", help="Also benchmark the PubMed-pretrained model before fine-tuning"
    )
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    outcome: dict[str, Any] = run_benchmarks(config, args.runs, args.pretrained)
    save_stage_result(config.results_dir, "benchmarks", outcome)
    print(json.dumps({run: value["summary"] for run, value in outcome["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
