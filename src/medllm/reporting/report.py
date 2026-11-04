"""Stage results of the pipeline and the comparison report.

Every DVC stage writes a small JSON file to `results/<model>/<stage>.json`
(run ids, model URIs, metrics, and each test answer marked right or wrong);
the next stages read it instead of searching MLflow, and DVC tracks it as the
stage output. The models themselves live in the MLflow Model Registry.

The report compares the zero-shot model with every fine-tuning run of the
given base models. Accuracies come with 95% bootstrap confidence intervals
over the test questions, and every fine-tuning run is compared with the
zero-shot model on the same questions (paired bootstrap).

    uv run python -m medllm.reporting.report --models qwen3-1.7b
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BASES = ("original", "dapt")
ZERO_SHOT = "zero-shot"
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10_000
# Resamples drawn at once (bounds the memory of the index matrix)
BOOTSTRAP_CHUNK = 1_000

# Choice tasks: answers marked right or wrong, so accuracy gets a confidence interval
CHOICE_TASKS: dict[str, str] = {"pubmedqa": "PubMedQA", "medmcqa": "MedMCQA", "medqa": "MedQA"}
GEVAL_CRITERIA = ("correctness", "completeness", "coherence")

# Metric key and column title, in report order
REPORT_METRICS: dict[str, str] = {
    "test_medtext_loss": "MedText loss",
    "test_pubmedqa_accuracy": "PubMedQA acc",
    "test_pubmedqa_macro_f1": "PubMedQA F1",
    "test_medmcqa_accuracy": "MedMCQA acc",
    "test_medmcqa_macro_f1": "MedMCQA F1",
    "test_medqa_accuracy": "MedQA acc",
    "test_medqa_macro_f1": "MedQA F1",
    "test_medtext_geval_correctness": "G-Eval correctness",
    "test_medtext_geval_completeness": "G-Eval completeness",
    "test_medtext_geval_coherence": "G-Eval coherence",
}


def save_stage_result(results_dir: str | Path, stage: str, result: dict[str, Any]) -> Path:
    """Write a stage result to `<results_dir>/<stage>.json`.

    Returns:
        The path written.
    """
    path = Path(results_dir) / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def load_stage_result(results_dir: str | Path, stage: str) -> dict[str, Any]:
    """Read the result of an earlier stage.

    Raises:
        FileNotFoundError: If that stage has not run for this model yet.
    """
    path = Path(results_dir) / f"{stage}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the {stage} stage first")
    return json.loads(path.read_text())


def resampled_means(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Means of `n_resamples` bootstrap resamples (drawn with replacement) of `values`."""
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, n_resamples, BOOTSTRAP_CHUNK):
        size: int = min(BOOTSTRAP_CHUNK, n_resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means.append(values[indices].mean(axis=1))
    return np.concatenate(means)


def bootstrap_ci(
    values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval (`CONFIDENCE_LEVEL`) of the mean."""
    means = resampled_means(np.asarray(values, dtype=float), n_resamples, seed)
    tail: float = (1 - CONFIDENCE_LEVEL) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_bootstrap(
    before: list[float], after: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> dict[str, float]:
    """Difference of two models scored on the same examples, resampling the examples.

    Args:
        before: Per-example scores of the reference model (1/0 for right/wrong).
        after: Per-example scores of the compared model, in the same order.

    Returns:
        `delta` (mean of after - before), its confidence interval `low`-`high`,
        and a two-sided `p_value` (share of resamples on the other side of zero).

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(before) != len(after):
        raise ValueError(f"Paired scores need the same examples: {len(before)} vs {len(after)}")

    differences = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    delta = float(differences.mean())
    if not differences.any():
        return {"delta": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}

    means = resampled_means(differences, n_resamples, seed)
    tail: float = (1 - CONFIDENCE_LEVEL) / 2
    p_value: float = min(1.0, 2 * min(float((means <= 0).mean()), float((means >= 0).mean())))
    return {
        "delta": delta,
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1 - tail)),
        "p_value": p_value,
    }


def majority_accuracy(label_counts: dict[str, int]) -> float:
    """Accuracy of always answering the most common label."""
    return max(label_counts.values()) / sum(label_counts.values())


def mean_over_seeds(seeds: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per task, each test answer's correctness averaged over the seeds (empty if not recorded)."""
    with_answers: list[dict[str, list[int]]] = [seed["correct"] for seed in seeds if "correct" in seed]
    if not with_answers:
        return {}
    return {
        task: [float(np.mean(values)) for values in zip(*(answers[task] for answers in with_answers))]
        for task in with_answers[0]
    }


def run_row(model: str, run: str, result: dict[str, Any]) -> dict[str, Any]:
    """Report row of one stage result: its summary metrics and per-answer correctness."""
    row: dict[str, Any] = {"model": model, "run": run, "geval_scores": {}}
    row.update(result["summary"])
    row["correct"] = mean_over_seeds(result.get("seeds", []))
    return row


def comparison_rows(results_root: str | Path, models: list[str]) -> list[dict[str, Any]]:
    """Zero-shot row, then one row per fine-tuning run, for each base model.

    Args:
        results_root: Directory with one `results/<model>/` directory per base model.
        models: Base models to include, in report order.

    Returns:
        Rows with `model`, `run`, `{metric: {"mean", "std"}}`, `correct`
        (per task, the correctness of each test answer averaged over seeds)
        and `geval_scores` (per criterion, the score of each MedText answer).
    """
    rows: list[dict[str, Any]] = []
    for model in models:
        model_dir = Path(results_root) / model
        geval: dict[str, Any] = {}
        if (model_dir / "geval.json").exists():
            geval = json.loads((model_dir / "geval.json").read_text())

        if (model_dir / "baseline.json").exists():
            rows.append(run_row(model, ZERO_SHOT, json.loads((model_dir / "baseline.json").read_text())))

        for path in sorted(model_dir.glob("finetune_*.json")):
            run: str = path.stem.removeprefix("finetune_")
            row = run_row(model, run, json.loads(path.read_text()))
            row.update(geval.get("bases", {}).get(run, {}))
            row["geval_scores"] = geval.get("scores", {}).get(run, {})
            rows.append(row)
    return rows


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Accuracy or G-Eval score with its confidence interval when per-example values exist, else mean ± std."""
    if metric not in row:
        return "-"
    mean: float = row[metric]["mean"]

    values: list[float] | None = None
    for task in CHOICE_TASKS:
        if metric == f"test_{task}_accuracy":
            values = row["correct"].get(task)
    for criterion in GEVAL_CRITERIA:
        if metric == f"test_medtext_geval_{criterion}":
            values = row["geval_scores"].get(criterion)

    if values:
        low, high = bootstrap_ci(values)
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"
    return f"{mean:.3f} ± {row[metric]['std']:.3f}"


def comparison_markdown(rows: list[dict[str, Any]], majority: dict[str, float]) -> str:
    """Markdown table of `REPORT_METRICS` per row, plus the accuracy of guessing.

    Args:
        rows: Rows from `comparison_rows`.
        majority: Accuracy without any knowledge, per choice task (see `guessing_accuracy`).
    """
    header = ["Model", "Run", *REPORT_METRICS.values()]
    lines: list[str] = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        cells: list[str] = [row["model"], row["run"]]
        cells.extend(metric_cell(row, metric) for metric in REPORT_METRICS)
        lines.append("| " + " | ".join(cells) + " |")

    majority_cells: list[str] = ["", "chance / majority class"]
    for metric in REPORT_METRICS:
        task: str = metric.removeprefix("test_").removesuffix("_accuracy")
        majority_cells.append(f"{majority[task]:.3f}" if metric.endswith("_accuracy") and task in majority else "")
    lines.append("| " + " | ".join(majority_cells) + " |")
    return "\n".join(lines) + "\n"


def paired_comparisons_markdown(rows: list[dict[str, Any]]) -> str:
    """Markdown table: every fine-tuning run against the zero-shot model of the same base model."""
    lines: list[str] = [
        "| Model | Run | Task | Δ accuracy vs zero-shot [95% CI] | p-value |",
        "|---|---|---|---|---|",
    ]
    zero_shot: dict[str, dict[str, Any]] = {row["model"]: row for row in rows if row["run"] == ZERO_SHOT}
    for row in rows:
        reference = zero_shot.get(row["model"])
        if row["run"] == ZERO_SHOT or reference is None:
            continue
        for task, title in CHOICE_TASKS.items():
            if task not in row["correct"] or task not in reference["correct"]:
                continue
            result = paired_bootstrap(reference["correct"][task], row["correct"][task])
            p_value: str = "<0.001" if result["p_value"] < 0.001 else f"{result['p_value']:.3f}"
            lines.append(
                f"| {row['model']} | {row['run']} | {title} | "
                f"{result['delta']:+.3f} [{result['low']:+.3f}, {result['high']:+.3f}] | {p_value} |"
            )
    return "\n".join(lines) + "\n"


def guessing_accuracy(stats_path: str | Path) -> dict[str, float]:
    """Accuracy without any knowledge, per choice task.

    PubMedQA: always answering its most common decision (from the `stats.json`
    of the multitask data). MedMCQA and MedQA are asked open-ended, so the
    model never sees the option letters: a random option is right 1 time in 4.
    """
    stats: dict[str, Any] = json.loads(Path(stats_path).read_text())
    return {
        "pubmedqa": majority_accuracy(stats["test_labels"]["test_pubmedqa"]),
        "medmcqa": 0.25,
        "medqa": 0.25,
    }


def benchmarks_markdown(results_root: str | Path, models: list[str], group: str = "tasks") -> str:
    """Table of the lm-evaluation-harness accuracies: mean over seeds ± standard error, and their average.

    Args:
        results_root: Directory with one results directory per model.
        models: Models to include (those without `benchmarks.json` are skipped).
        group: Tasks to show: `tasks` (medical) or `general_tasks`.

    Returns:
        A Markdown table, or an empty string when no model has results for the group.
    """
    lines: list[str] = []
    for model in models:
        path = Path(results_root) / model / "benchmarks.json"
        if not path.exists():
            continue
        benchmarks: dict[str, Any] = json.loads(path.read_text())
        if group not in benchmarks:
            continue
        tasks: list[str] = benchmarks[group]
        if not lines:
            lines.append("| Model | Run | " + " | ".join(tasks) + " | Average |")
            lines.append("|" + "---|" * (len(tasks) + 3))
        for run, result in benchmarks["runs"].items():
            seeds: list[dict[str, float]] = [seed["metrics"] for seed in result["seeds"]]
            accuracies: list[float] = []
            cells: list[str] = []
            for task in tasks:
                accuracy: float = float(np.mean([metrics[f"benchmark_{task}_acc"] for metrics in seeds]))
                stderr: float = float(np.mean([metrics[f"benchmark_{task}_acc_stderr"] for metrics in seeds]))
                accuracies.append(accuracy)
                cells.append(f"{accuracy:.3f} ± {stderr:.3f}")
            lines.append(f"| {model} | {run} | " + " | ".join(cells) + f" | {np.mean(accuracies):.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the zero-shot and fine-tuned models.")
    parser.add_argument("--models", nargs="+", required=True, help="Directories under --results-dir")
    parser.add_argument("--data-stats", required=True, help="stats.json of the multitask data (test labels)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="reports/comparison.md")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = comparison_rows(args.results_dir, args.models)
    table: str = comparison_markdown(rows, guessing_accuracy(args.data_stats))
    paired: str = paired_comparisons_markdown(rows)
    benchmarks: str = benchmarks_markdown(args.results_dir, args.models)
    general_benchmarks: str = benchmarks_markdown(args.results_dir, args.models, "general_tasks")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Model comparison\n\n"
        "Test metrics. Accuracies and G-Eval scores: mean over seeds with a 95% bootstrap "
        "confidence interval over the test questions; other metrics: mean ± std over seeds. "
        "`zero-shot` is the model without fine-tuning; runs are named `<method>_<start>`, "
        "where `original` starts from the Hub model and `dapt` from the PubMed-pretrained one.\n\n"
        + table
        + "\n## Fine-tuning vs zero-shot\n\n"
        "Paired bootstrap over the same test questions.\n\n"
        + paired
        + "\n## Standard benchmarks\n\n"
        "lm-evaluation-harness, 0-shot, multiple choice with the options shown, in the chat "
        "format: accuracy ± standard error. Comparable with published results. "
        "`zero-shot_dapt` is the PubMed-pretrained model before fine-tuning.\n\n"
        + (benchmarks or "Not run.")
        + "\n\n## General knowledge\n\n"
        "The same, on general tasks: a drop from `zero-shot` means the medical training "
        "made the model forget general knowledge.\n\n"
        + (general_benchmarks or "Not run.")
        + "\n"
    )
    print(table)
    print(paired)


if __name__ == "__main__":
    main()
