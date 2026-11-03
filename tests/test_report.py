from pathlib import Path

import pytest

from src.report import comparison_markdown, comparison_rows, load_stage_result, save_stage_result


def summary(value: float) -> dict[str, float]:
    return {"mean": value, "std": 0.01}


def test_save_stage_result_writes_json_that_load_reads_back(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "gpt2"

    path = save_stage_result(results_dir, "pretrain", {"run_id": "abc", "eval_loss": 2.5})

    assert path == results_dir / "pretrain.json"
    assert load_stage_result(results_dir, "pretrain") == {"run_id": "abc", "eval_loss": 2.5}


def test_load_stage_result_names_the_missing_stage(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="finetune_dapt"):
        load_stage_result(tmp_path, "finetune_dapt")


def test_comparison_rows_has_one_row_per_model_and_base(tmp_path: Path) -> None:
    for model in ("gpt2", "qwen3-1.7b"):
        for base in ("original", "dapt"):
            save_stage_result(
                tmp_path / model, f"finetune_{base}", {"summary": {"test_pubmedqa_accuracy": summary(0.6)}}
            )
    save_stage_result(
        tmp_path / "gpt2",
        "geval",
        {"bases": {"dapt": {"test_medtext_geval_correctness": summary(0.1)}}},
    )

    rows = comparison_rows(tmp_path)

    assert [(row["model"], row["base"]) for row in rows] == [
        ("gpt2", "original"),
        ("gpt2", "dapt"),
        ("qwen3-1.7b", "original"),
        ("qwen3-1.7b", "dapt"),
    ]
    assert rows[1]["test_pubmedqa_accuracy"] == summary(0.6)
    assert rows[1]["test_medtext_geval_correctness"] == summary(0.1)
    assert "test_medtext_geval_correctness" not in rows[0]


def test_comparison_markdown_shows_mean_and_std_and_blanks_missing_metrics() -> None:
    rows = [
        {"model": "gpt2", "base": "original", "test_pubmedqa_accuracy": summary(0.6381)},
    ]

    table = comparison_markdown(rows)

    lines = table.splitlines()
    assert lines[0].startswith("| Model | Base |")
    assert "| gpt2 | original |" in lines[2]
    assert "0.638 ± 0.010" in lines[2]
    assert "| - |" in lines[2]
