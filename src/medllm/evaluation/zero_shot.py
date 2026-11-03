"""Zero-shot baseline: the base model scored on the test sets without fine-tuning.

    uv run python -m medllm.evaluation.zero_shot model=qwen3-1.7b

Writes `results/<model>/baseline.json`, the reference every fine-tuning run is
compared with in the report.
"""

import argparse
import json
from typing import Any

import mlflow

from medllm.config import MultitaskConfig, load_multitask_config
from medllm.evaluation.tasks import TaskEvaluation, evaluate_model
from medllm.modeling import model_label
from medllm.reporting.report import save_stage_result
from medllm.tracking import configure_tracking, start_stage_run


def run_baseline(config: MultitaskConfig) -> dict[str, Any]:
    """Score the model without fine-tuning (zero-shot) on the test sets, as an `evaluate` run.

    Returns:
        The model, the run id, its test answers marked right or wrong (as a
        single "seed", like the fine-tuning results) and its metrics.
    """
    model_id: str = config.model.model_id
    configure_tracking(config)
    with start_stage_run(
        "evaluate",
        f"zero-shot-{model_label(model_id)}",
        config,
        {"task": "multitask", "evaluated_model": model_id, "split": "test"},
    ) as run:
        evaluation: TaskEvaluation = evaluate_model(model_id, config, "test")
        mlflow.log_metrics(evaluation.metrics)

    return {
        "model_id": model_id,
        "run_id": run.info.run_id,
        "seeds": [{"seed": None, "run_id": run.info.run_id, "correct": evaluation.correct}],
        "summary": {key: {"mean": value, "std": 0.0} for key, value in evaluation.metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the base model without fine-tuning (zero-shot).")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example model=qwen3-1.7b")
    args = parser.parse_args()

    config: MultitaskConfig = load_multitask_config(args.overrides)
    outcome: dict[str, Any] = run_baseline(config)
    save_stage_result(config.results_dir, "baseline", outcome)
    print(json.dumps(outcome["summary"], indent=2))


if __name__ == "__main__":
    main()
