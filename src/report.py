"""Stage results of the pipeline and the comparison report across base models.

Every DVC stage writes a small JSON file to `results/<model>/<stage>.json`
(run ids, model URIs, metrics); the next stages read it instead of searching
MLflow, and DVC tracks it as the stage output. The models themselves live in
the MLflow Model Registry.

    uv run python -m src.report
"""

import argparse
import json
from pathlib import Path
from typing import Any

BASES = ("original", "dapt")

# Metric key and column title, in report order
REPORT_METRICS: dict[str, str] = {
    "test_medtext_loss": "MedText loss",
    "test_pubmedqa_accuracy": "PubMedQA acc",
    "test_pubmedqa_macro_f1": "PubMedQA F1",
    "test_medmcqa_accuracy": "MedMCQA acc",
    "test_medmcqa_macro_f1": "MedMCQA F1",
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


def comparison_rows(results_root: str | Path) -> list[dict[str, Any]]:
    """One row per base model and starting point (original or dapt) with a fine-tuning result.

    Args:
        results_root: Directory with one `results/<model>/` directory per base model.

    Returns:
        Rows with `model`, `base` and `{metric: {"mean", "std"}}` from the
        fine-tuning summary and, when present, the G-Eval summary.
    """
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(Path(results_root).iterdir()):
        if not model_dir.is_dir():
            continue

        geval_path = model_dir / "geval.json"
        geval_by_base: dict[str, Any] = {}
        if geval_path.exists():
            geval_by_base = json.loads(geval_path.read_text())["bases"]

        for base in BASES:
            finetune_path = model_dir / f"finetune_{base}.json"
            if not finetune_path.exists():
                continue
            row: dict[str, Any] = {"model": model_dir.name, "base": base}
            row.update(json.loads(finetune_path.read_text())["summary"])
            row.update(geval_by_base.get(base, {}))
            rows.append(row)
    return rows


def comparison_markdown(rows: list[dict[str, Any]]) -> str:
    """Markdown table of `REPORT_METRICS` as mean ± std over seeds ("-" when missing)."""
    header = ["Model", "Base", *REPORT_METRICS.values()]
    lines: list[str] = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        cells: list[str] = [row["model"], row["base"]]
        for metric in REPORT_METRICS:
            if metric in row:
                cells.append(f"{row[metric]['mean']:.3f} ± {row[metric]['std']:.3f}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the fine-tuned models of every base model.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="reports/comparison.md")
    args = parser.parse_args()

    table: str = comparison_markdown(comparison_rows(args.results_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Base model comparison\n\n"
        "Test metrics of the multitask fine-tuned models, mean ± std over seeds. "
        "`original` starts from the Hub model, `dapt` from the PubMed-pretrained one.\n\n"
        + table
    )
    print(table)


if __name__ == "__main__":
    main()
