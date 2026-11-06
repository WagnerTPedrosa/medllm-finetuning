"""Zero-shot baseline: a model scored on the test sets without fine-tuning.

    uv run python -m medllm.evaluation.zero_shot model=qwen3-1.7b
    uv run python -m medllm.evaluation.zero_shot model=qwen3-1.7b --base dapt

`--base original` (the default) scores the Hub model and writes
`results/<model>/baseline.json`, the reference every fine-tuning run is
compared with in the report. `--base dapt` scores the PubMed-pretrained model
(`finetune.dapt_model_id`) and writes `baseline_dapt.json`: what the
pretraining alone does to the answers.
"""

import argparse
import json
from typing import Any

import mlflow

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.evaluation.tasks import TaskEvaluation, evaluate_model
from medllm.modeling import model_label
from medllm.reporting.report import BASES, save_stage_result
from medllm.tracking import configure_tracking, sectioned_metrics, start_stage_run
from medllm.training.finetune import base_model_id


def baseline_stage(base: str) -> str:
    """Name of the stage result file: `baseline` for the Hub model, `baseline_dapt` for the pretrained one.

    Raises:
        ValueError: If `base` is not one of `BASES`.
    """
    if base not in BASES:
        raise ValueError(f"base must be one of {BASES}, got {base!r}")
    if base == "original":
        return "baseline"
    return f"baseline_{base}"


def run_baseline(config: MultitaskConfig, base: str = "original") -> dict[str, Any]:
    """Score a model without fine-tuning (zero-shot) on the test sets, as an `evaluate` run.

    Args:
        config: Multitask configuration.
        base: `original` (the Hub model) or `dapt` (the PubMed-pretrained one).

    Returns:
        The model, the run id, its test answers marked right or wrong (as a
        single "seed", like the fine-tuning results) and its metrics.
    """
    model_id: str = base_model_id(config, base)
    configure_tracking(config)
    with start_stage_run(
        "evaluate",
        f"zero-shot-{base}-{model_label(config.model.model_id)}",
        config,
        {"task": "multitask", "evaluated_model": model_id, "finetune_base": base, "split": "test"},
    ) as run:
        evaluation: TaskEvaluation = evaluate_model(model_id, config, "test")
        mlflow.log_metrics(sectioned_metrics(evaluation.metrics))

    return {
        "model_id": model_id,
        "run_id": run.info.run_id,
        "seeds": [{"seed": None, "run_id": run.info.run_id, "correct": evaluation.correct}],
        "summary": {key: {"mean": value, "std": 0.0} for key, value in evaluation.metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a model without fine-tuning (zero-shot).")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    parser.add_argument("--base", choices=BASES, default="original", help="Hub model or PubMed-pretrained model")
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    outcome: dict[str, Any] = run_baseline(config, args.base)
    save_stage_result(config.results_dir, baseline_stage(args.base), outcome)
    print(json.dumps(outcome["summary"], indent=2))


if __name__ == "__main__":
    main()
