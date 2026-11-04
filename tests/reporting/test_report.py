from pathlib import Path

import pytest

from medllm.reporting.report import (
    benchmarks_markdown,
    bootstrap_ci,
    comparison_markdown,
    comparison_rows,
    load_stage_result,
    majority_accuracy,
    paired_bootstrap,
    paired_comparisons_markdown,
    save_stage_result,
)


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


def test_bootstrap_ci_of_constant_values_is_that_value() -> None:
    assert bootstrap_ci([1, 1, 1, 1]) == (1.0, 1.0)


def test_bootstrap_ci_contains_the_mean() -> None:
    values = [1, 0] * 50

    low, high = bootstrap_ci(values)

    assert low < 0.5 < high
    assert high - low < 0.25


def test_paired_bootstrap_finds_no_difference_between_equal_answers() -> None:
    answers = [1, 0, 1, 1, 0] * 20

    result = paired_bootstrap(answers, answers)

    assert result["delta"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_bootstrap_detects_a_clear_improvement() -> None:
    before = [0] * 80 + [1] * 20
    after = [1] * 80 + [1] * 20

    result = paired_bootstrap(before, after)

    assert result["delta"] == pytest.approx(0.8)
    assert result["low"] > 0
    assert result["p_value"] < 0.001


def test_majority_accuracy_is_the_share_of_the_most_common_label() -> None:
    assert majority_accuracy({"A": 30, "B": 50, "C": 20}) == 0.5


def write_run(directory: Path, stage: str, accuracy: float, correct: list[int]) -> None:
    save_stage_result(
        directory,
        stage,
        {
            "seeds": [{"seed": 42, "correct": {"medqa": correct}}],
            "summary": {"test_medqa_accuracy": summary(accuracy)},
        },
    )


def test_comparison_rows_puts_zero_shot_first_then_each_fine_tuning(tmp_path: Path) -> None:
    write_run(tmp_path / "qwen3-1.7b", "baseline", 0.4, [0, 1, 0, 1, 0])
    write_run(tmp_path / "qwen3-1.7b", "finetune_lora_original", 0.6, [1, 1, 0, 1, 0])
    save_stage_result(
        tmp_path / "qwen3-1.7b",
        "geval",
        {
            "bases": {"lora_original": {"test_medtext_geval_correctness": summary(0.3)}},
            "scores": {"lora_original": {"correctness": [0.2, 0.4]}},
        },
    )

    rows = comparison_rows(tmp_path, ["qwen3-1.7b"])

    assert [(row["model"], row["run"]) for row in rows] == [
        ("qwen3-1.7b", "zero-shot"),
        ("qwen3-1.7b", "lora_original"),
    ]
    assert rows[1]["correct"]["medqa"] == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert rows[1]["geval_scores"]["correctness"] == [0.2, 0.4]
    assert rows[1]["test_medtext_geval_correctness"] == summary(0.3)


def test_comparison_rows_averages_correctness_over_seeds(tmp_path: Path) -> None:
    save_stage_result(
        tmp_path / "m",
        "finetune_full_original",
        {
            "seeds": [{"correct": {"medqa": [1, 0]}}, {"correct": {"medqa": [1, 1]}}],
            "summary": {},
        },
    )

    rows = comparison_rows(tmp_path, ["m"])

    assert rows[0]["correct"]["medqa"] == [1.0, 0.5]


def test_comparison_markdown_shows_confidence_intervals_and_blanks_missing_metrics() -> None:
    rows = [
        {
            "model": "qwen3-1.7b",
            "run": "lora_original",
            "test_medqa_accuracy": summary(0.6),
            "correct": {"medqa": [1, 0] * 50},
            "geval_scores": {},
        }
    ]

    table = comparison_markdown(rows, majority={"medqa": 0.28})

    lines = table.splitlines()
    assert lines[0].startswith("| Model | Run |")
    assert "| qwen3-1.7b | lora_original |" in lines[2]
    assert "0.600 [" in lines[2]
    assert "| - |" in lines[2]
    assert "| chance / majority class |" in lines[3]
    assert "0.280" in lines[3]


def test_paired_comparisons_markdown_compares_each_run_with_zero_shot() -> None:
    rows = [
        {"model": "m", "run": "zero-shot", "correct": {"medqa": [0] * 50 + [1] * 50}, "geval_scores": {}},
        {"model": "m", "run": "lora_original", "correct": {"medqa": [1] * 100}, "geval_scores": {}},
    ]

    table = paired_comparisons_markdown(rows)

    assert "| m | lora_original | MedQA | +0.500 [" in table
    assert "<0.001" in table


def benchmark_seed(medqa: float, pubmedqa: float) -> dict[str, dict[str, float]]:
    return {
        "metrics": {
            "benchmark_medqa_4options_acc": medqa,
            "benchmark_medqa_4options_acc_stderr": 0.01,
            "benchmark_pubmedqa_test_acc": pubmedqa,
            "benchmark_pubmedqa_test_acc_stderr": 0.02,
        }
    }


def test_benchmarks_markdown_shows_accuracy_with_standard_error_and_the_average(tmp_path: Path) -> None:
    save_stage_result(
        tmp_path / "m",
        "benchmarks",
        {
            "tasks": ["medqa_4options", "pubmedqa_test"],
            "runs": {
                "zero-shot": {"seeds": [benchmark_seed(0.4, 0.6)]},
                "lora_original": {"seeds": [benchmark_seed(0.5, 0.7)]},
            },
        },
    )

    table = benchmarks_markdown(tmp_path, ["m"])

    lines = table.splitlines()
    assert lines[0] == "| Model | Run | medqa_4options | pubmedqa_test | Average |"
    assert lines[2] == "| m | zero-shot | 0.400 ± 0.010 | 0.600 ± 0.020 | 0.500 |"
    assert lines[3] == "| m | lora_original | 0.500 ± 0.010 | 0.700 ± 0.020 | 0.600 |"


def test_benchmarks_markdown_is_empty_without_results(tmp_path: Path) -> None:
    assert benchmarks_markdown(tmp_path, ["m"]) == ""
