"""LLM-as-judge evaluation (G-Eval) of the answers generated for the MedText test set.

The fine-tuned models generate an answer for every MedText test question and a
local judge model, served by vLLM, scores it against the reference answer on
three criteria (correctness, completeness, coherence). Scores are logged to the
seed runs of the latest `multitask-<base>` runs, and their mean and standard
deviation to the parent runs, next to the other test metrics.

Run with the vLLM server up (see the `geval` section of the config):

    uv run python -m src.geval --config configs/gpt2_multitask.yaml
"""

import os

# DeepEval sends usage telemetry by default; nothing should leave this machine.
# The variable must be set before deepeval is imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from datasets import Dataset
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM, LocalModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from mlflow.entities import Metric, Run
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.config import GEvalConfig, MultitaskConfig, load_multitask_config
from src.inference import generate_completion
from src.modeling import load_model
from src.multitask import load_multitask
from src.multitask_train import model_label, summarize_seeds
from src.tracking import configure_tracking

METRIC_PREFIX = "test_medtext_geval"

REFERENCE_PARAMS: list[SingleTurnParams] = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
]

# Evaluation steps per criterion. Fixed steps (instead of a one-line criterion)
# make the judge more consistent across runs.
GEVAL_METRICS: dict[str, tuple[list[str], list[SingleTurnParams]]] = {
    "correctness": (
        [
            "Identify the main clinical conclusion of the expected output: the diagnosis, "
            "explanation or recommended action it gives.",
            "Check whether the actual output reaches the same main conclusion. A different or "
            "missing main conclusion must get a low score.",
            "Heavily penalize statements in the actual output that contradict the expected output "
            "or established medical knowledge, including invented facts.",
            "Do not penalize different wording, or extra details that are medically correct.",
        ],
        REFERENCE_PARAMS,
    ),
    "completeness": (
        [
            "List the key clinical points of the expected output, for example the cause, "
            "diagnosis, tests to order, treatment and urgency.",
            "Check which of these points the actual output covers correctly.",
            "Score by the fraction of key points covered. Ignore style and length.",
        ],
        REFERENCE_PARAMS,
    ),
    "coherence": (
        [
            "Check whether the actual output is a direct answer to the kind of question asked "
            "(for example it gives a diagnosis or a next step when asked for one).",
            "Penalize self-contradictions, circular statements that only restate the question, "
            "and repeated sentences.",
            "Do not judge medical accuracy: a wrong diagnosis or treatment stated clearly and "
            "consistently must still get a high score. Accuracy is scored by another metric.",
        ],
        [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    ),
}


def build_judge(geval_config: GEvalConfig) -> LocalModel:
    """Judge model behind an OpenAI-compatible server (vLLM), with greedy decoding.

    Args:
        geval_config: Judge model name and server URL.

    Returns:
        A DeepEval model that sends its requests to the server.
    """
    return LocalModel(
        model=geval_config.judge_model,
        base_url=geval_config.base_url,
        api_key="EMPTY",  # vLLM accepts any key unless it was started with --api-key
        temperature=0.0,
        # Qwen3 thinks before answering by default; the judge only needs the JSON verdict
        generation_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )


def build_geval_metrics(judge: DeepEvalBaseLLM) -> list[GEval]:
    """One G-Eval metric per criterion in `GEVAL_METRICS`, in the same order.

    Args:
        judge: Model that scores the answers.

    Returns:
        New metric objects (they keep the last score, so use one set per test case).
    """
    return [
        GEval(
            name=name,
            evaluation_steps=steps,
            evaluation_params=params,
            model=judge,
            async_mode=True,
        )
        for name, (steps, params) in GEVAL_METRICS.items()
    ]


def build_test_cases(
    prompts: list[str], answers: list[str], references: list[str]
) -> list[LLMTestCase]:
    """Pair every prompt with the generated answer and the reference answer.

    Raises:
        ValueError: If the three lists do not have the same length.
    """
    if not len(prompts) == len(answers) == len(references):
        raise ValueError(
            f"Got {len(prompts)} prompts, {len(answers)} answers and {len(references)} references"
        )

    return [
        LLMTestCase(input=prompt, actual_output=answer, expected_output=reference)
        for prompt, answer, reference in zip(prompts, answers, references)
    ]


async def score_test_case(
    test_case: LLMTestCase, judge: DeepEvalBaseLLM, semaphore: asyncio.Semaphore
) -> dict[str, float | str]:
    """Score one answer on every criterion.

    Returns:
        `{criterion: score in [0, 1], f"{criterion}_reason": judge's explanation}`.
    """
    row: dict[str, float | str] = {}
    async with semaphore:
        for name, metric in zip(GEVAL_METRICS, build_geval_metrics(judge)):
            await metric.a_measure(test_case, _show_indicator=False)
            row[name] = metric.score
            row[f"{name}_reason"] = metric.reason
    return row


async def score_test_cases(
    test_cases: list[LLMTestCase], judge: DeepEvalBaseLLM, max_concurrent: int
) -> list[dict[str, float | str]]:
    """Score all answers, with at most `max_concurrent` test cases in flight.

    Returns:
        One row per test case, in the same order (see `score_test_case`).
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    return await asyncio.gather(
        *(score_test_case(test_case, judge, semaphore) for test_case in test_cases)
    )


def summarize_geval(
    rows: list[dict[str, float | str]], metric_names: list[str], prefix: str
) -> dict[str, float]:
    """Mean score of each criterion, named `<prefix>_<criterion>`."""
    return {
        f"{prefix}_{name}": sum(float(row[name]) for row in rows) / len(rows)
        for name in metric_names
    }


def seed_model_dir(config: MultitaskConfig, base_model: str, seed: str) -> Path:
    """Directory where `train_multitask` saved the best epoch of a seed."""
    return Path(config.finetune.output_dir) / f"{model_label(base_model)}-seed{seed}" / "best"


def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    """Greedy answer to every prompt."""
    answers: list[str] = []
    for prompt in prompts:
        answers.append(generate_completion(model, tokenizer, prompt, device, max_new_tokens))
    return answers


def latest_multitask_runs(config: MultitaskConfig) -> dict[str, list[Run]]:
    """Seed runs of the latest finished `multitask-<base>` run of each base model.

    Returns:
        `{parent run id: its seed runs}`.
    """
    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    parents: list[Run] = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string="tags.task = 'multitask' AND tags.stage = 'finetune' AND attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        output_format="list",
    )

    seed_runs: dict[str, list[Run]] = {}
    seen_bases: set[str] = set()
    for parent in parents:
        base: str = parent.data.tags["base_model"]
        if "finetune_seed" in parent.data.tags or base in seen_bases:
            continue
        seen_bases.add(base)
        seed_runs[parent.info.run_id] = mlflow.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            output_format="list",
        )
    return seed_runs


def evaluate_seed_run(
    run: Run, config: MultitaskConfig, judge: DeepEvalBaseLLM, test_set: Dataset
) -> dict[str, float]:
    """Generate answers with a seed's best model, score them and log the results to its run.

    Logs the mean score per criterion as metrics and every answer with the
    judge's scores and reasons as the `geval/test_medtext.json` table.

    Returns:
        The mean score per criterion.
    """
    tags: dict[str, str] = run.data.tags
    model_dir: Path = seed_model_dir(config, tags["base_model"], tags["finetune_seed"])
    print(f"Generating answers with {model_dir}")

    model, device = load_model(str(model_dir))
    model.config.use_cache = True  # load_model turns the cache off for training
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
    answers: list[str] = generate_answers(
        model, tokenizer, test_set["prompt"], device, config.geval.max_new_tokens
    )
    del model
    torch.cuda.empty_cache()

    print(f"Scoring {len(answers)} answers with {config.geval.judge_model}")
    test_cases = build_test_cases(test_set["prompt"], answers, test_set["completion"])
    rows = asyncio.run(score_test_cases(test_cases, judge, config.geval.max_concurrent))
    summary: dict[str, float] = summarize_geval(rows, list(GEVAL_METRICS), METRIC_PREFIX)

    run_id: str = run.info.run_id
    client = mlflow.MlflowClient()
    client.log_batch(run_id, metrics=[Metric(key, value, 0, 0) for key, value in summary.items()])
    client.set_tag(run_id, "geval_judge", config.geval.judge_model)
    table = pd.DataFrame(
        {"prompt": test_set["prompt"], "reference": test_set["completion"], "answer": answers}
    ).join(pd.DataFrame(rows))
    mlflow.log_table(table, "geval/test_medtext.json", run_id=run_id)
    return summary


def run_geval(config: MultitaskConfig) -> dict[str, dict[str, dict[str, float]]]:
    """Score every seed of the latest multitask runs and log the mean and std per base model.

    Returns:
        `{base model: {metric: {"mean": ..., "std": ...}}}`.
    """
    configure_tracking(config)
    test_set: Dataset = load_multitask(config.multitask.processed_dir)["test_medtext"]
    if config.geval.num_examples is not None:
        test_set = test_set.select(range(min(config.geval.num_examples, len(test_set))))
    judge = build_judge(config.geval)

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    client = mlflow.MlflowClient()
    for parent_id, seed_runs in latest_multitask_runs(config).items():
        results: list[dict[str, float]] = [
            evaluate_seed_run(run, config, judge, test_set) for run in seed_runs
        ]
        summary = summarize_seeds(results)
        metrics: list[Metric] = []
        for key, value in summary.items():
            metrics.append(Metric(f"{key}_mean", value["mean"], 0, 0))
            metrics.append(Metric(f"{key}_std", value["std"], 0, 0))
        client.log_batch(parent_id, metrics=metrics)
        client.set_tag(parent_id, "geval_judge", config.geval.judge_model)
        summaries[client.get_run(parent_id).data.tags["base_model"]] = summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="G-Eval of the multitask models on the MedText test set.")
    parser.add_argument("--config", default="configs/gpt2_multitask.yaml")
    args = parser.parse_args()

    summaries = run_geval(load_multitask_config(args.config))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
