"""LLM-as-judge evaluation (G-Eval) of the answers generated for the MedText test set.

Every seed of the given fine-tuning runs (`results/<model>/finetune_<run>.json`)
answers the MedText test questions. Then a local judge model, served by vLLM,
scores each answer against the reference on three criteria (correctness,
completeness, coherence). The answers are generated first, so the judge gets
the GPU to itself. Scores are logged to the seed runs and their mean and
standard deviation to the parent runs, next to the other test metrics.

    uv run python -m medllm.evaluation.geval --runs lora_original

The stage starts the vLLM server and stops it at the end, unless a server is
already listening at `geval.base_url` (see the `geval` section of the config).
"""

import os

# DeepEval sends usage telemetry by default; nothing should leave this machine.
# The variable must be set before deepeval is imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import argparse
import asyncio
import gc
import json
import signal
import statistics
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import mlflow
import pandas as pd
import torch
from datasets import Dataset
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM, LocalModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from mlflow.entities import Metric
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from medllm.chat import Message, message_text, prompt_text
from medllm.config import GEvalConfig, MultitaskConfig, load_multitask_config
from medllm.data.finetune_data import load_finetune_data
from medllm.evaluation.generation import generate_completion
from medllm.modeling import load_model
from medllm.reporting.report import load_stage_result, save_stage_result
from medllm.tracking import configure_tracking
from medllm.training.finetune import summarize_seeds

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


def build_serve_command(geval_config: GEvalConfig) -> list[str]:
    """`vllm serve` command for the judge, on the port of `base_url`."""
    port: int | None = urlparse(geval_config.base_url).port
    return [
        str(Path(geval_config.vllm_bin) / "vllm"),
        "serve",
        geval_config.judge_model,
        f"--port={port}",
        *geval_config.serve_args,
    ]


def server_is_up(base_url: str) -> bool:
    """Whether an OpenAI-compatible server answers at `base_url`."""
    try:
        with urlopen(f"{base_url}/models", timeout=2):
            return True
    except (URLError, ConnectionError, TimeoutError):
        return False


class JudgeServer:
    """Context manager that serves the judge with vLLM while the answers are scored.

    A server already listening at `base_url` is used as it is and left running.
    Otherwise vLLM is started (output in `server_log`) and stopped on exit.
    """

    def __init__(self, geval_config: GEvalConfig) -> None:
        self.config = geval_config
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "JudgeServer":
        if server_is_up(self.config.base_url):
            print(f"Using the judge server already running at {self.config.base_url}")
            return self

        log_path = Path(self.config.server_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log_file:
            # vLLM compiles kernels with the `ninja` installed next to it
            env: dict[str, str] = {**os.environ, "PATH": f"{self.config.vllm_bin}:{os.environ['PATH']}"}
            # Its own process group, so the engine processes vLLM starts are stopped with it
            self.process = subprocess.Popen(
                build_serve_command(self.config),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        print(f"Starting the judge server ({self.config.judge_model}), log in {log_path}")

        deadline: float = time.monotonic() + self.config.startup_timeout
        while not server_is_up(self.config.base_url):
            if self.process.poll() is not None:
                raise RuntimeError(f"The judge server exited with code {self.process.returncode}; see {log_path}")
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f"The judge server did not start in {self.config.startup_timeout} s; see {log_path}")
            time.sleep(5)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def stop(self) -> None:
        """Stop the server this object started (no-op for a server it did not start)."""
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()


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


def generate_seed_answers(model_dir: str, conversations: list[list[Message]], max_new_tokens: int) -> list[str]:
    """Answers of one fine-tuned model to chat prompts; the model is freed from the GPU afterwards."""
    print(f"Generating answers with {model_dir}")
    model, device = load_model(model_dir)
    model.config.use_cache = True  # load_model turns the cache off for training
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts: list[str] = [prompt_text(tokenizer, messages) for messages in conversations]
    answers: list[str] = generate_answers(model, tokenizer, prompts, device, max_new_tokens)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def log_seed_scores(
    run_id: str,
    questions: list[str],
    references: list[str],
    answers: list[str],
    rows: list[dict[str, float | str]],
    judge_model: str,
) -> dict[str, float]:
    """Log the mean score per criterion and the `geval/test_medtext.json` table to a seed run.

    Returns:
        The mean score per criterion.
    """
    summary: dict[str, float] = summarize_geval(rows, list(GEVAL_METRICS), METRIC_PREFIX)
    client = mlflow.MlflowClient()
    client.log_batch(run_id, metrics=[Metric(key, value, 0, 0) for key, value in summary.items()])
    client.set_tag(run_id, "geval_judge", judge_model)
    table = pd.DataFrame(
        {"question": questions, "reference": references, "answer": answers}
    ).join(pd.DataFrame(rows))
    mlflow.log_table(table, "geval/test_medtext.json", run_id=run_id)
    return summary


def mean_scores_per_example(
    seed_rows: list[list[dict[str, float | str]]], metric_names: list[str]
) -> dict[str, list[float]]:
    """Score of every answer averaged over seeds, per criterion (for the report's confidence intervals)."""
    return {
        name: [statistics.mean(float(rows[i][name]) for rows in seed_rows) for i in range(len(seed_rows[0]))]
        for name in metric_names
    }


def run_geval(config: MultitaskConfig, runs: list[str]) -> dict[str, Any]:
    """Score every seed of the given fine-tuning runs and log the mean and std per run.

    Args:
        config: Multitask configuration.
        runs: Fine-tuning results to score, by the suffix of their file
            (`lora_original` for `results/<model>/finetune_lora_original.json`).

    Returns:
        The judge model, `{run: {metric: {"mean": ..., "std": ...}}}` under
        `bases`, and the score of every answer per run and criterion under `scores`.
    """
    configure_tracking(config)
    test_set: Dataset = load_finetune_data(config.multitask.processed_dir)["test_medtext"]
    if config.geval.num_examples is not None:
        test_set = test_set.select(range(min(config.geval.num_examples, len(test_set))))
    questions: list[str] = [message_text(messages) for messages in test_set["prompt"]]
    references: list[str] = [message_text(messages) for messages in test_set["completion"]]
    finetunes: dict[str, dict[str, Any]] = {
        run: load_stage_result(config.results_dir, f"finetune_{run}") for run in runs
    }

    answers_by_run: dict[str, list[str]] = {}
    for finetune in finetunes.values():
        for seed in finetune["seeds"]:
            answers_by_run[seed["run_id"]] = generate_seed_answers(
                seed["model_dir"], test_set["prompt"], config.geval.max_new_tokens
            )

    judge = build_judge(config.geval)
    client = mlflow.MlflowClient()
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    scores: dict[str, dict[str, list[float]]] = {}
    with JudgeServer(config.geval):
        for run, finetune in finetunes.items():
            seed_summaries: list[dict[str, float]] = []
            seed_rows: list[list[dict[str, float | str]]] = []
            for seed in finetune["seeds"]:
                answers: list[str] = answers_by_run[seed["run_id"]]
                print(f"Scoring {len(answers)} answers of {run} seed {seed['seed']} with {config.geval.judge_model}")
                test_cases = build_test_cases(questions, answers, references)
                rows = asyncio.run(score_test_cases(test_cases, judge, config.geval.max_concurrent))
                seed_rows.append(rows)
                seed_summaries.append(
                    log_seed_scores(seed["run_id"], questions, references, answers, rows, config.geval.judge_model)
                )

            summary = summarize_seeds(seed_summaries)
            metrics: list[Metric] = []
            for key, value in summary.items():
                metrics.append(Metric(f"{key}_mean", value["mean"], 0, 0))
                metrics.append(Metric(f"{key}_std", value["std"], 0, 0))
            client.log_batch(finetune["parent_run_id"], metrics=metrics)
            client.set_tag(finetune["parent_run_id"], "geval_judge", config.geval.judge_model)
            summaries[run] = summary
            scores[run] = mean_scores_per_example(seed_rows, list(GEVAL_METRICS))

    return {"judge_model": config.geval.judge_model, "bases": summaries, "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description="G-Eval of the multitask models on the MedText test set.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Fine-tuning results to score, e.g. lora_original for results/<model>/finetune_lora_original.json",
    )
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    outcome: dict[str, Any] = run_geval(config, args.runs)
    save_stage_result(config.results_dir, "geval", outcome)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
