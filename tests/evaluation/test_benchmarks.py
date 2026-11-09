import pytest
from datasets import Dataset

from medllm.config import BenchmarksConfig, load_multitask_config
from medllm.evaluation.benchmarks import (
    benchmark_metrics,
    benchmark_tasks,
    harness_model_args,
    perplexity_metrics,
)
from medllm.evaluation.lm_eval_tasks.pubmedqa_test import doc_to_text, official_test


def test_benchmark_metrics_keeps_accuracy_and_its_standard_error_per_task() -> None:
    results = {
        "results": {
            "medqa_4options": {"alias": "medqa", "acc,none": 0.5, "acc_stderr,none": 0.01, "acc_norm,none": 0.52},
            "pubmedqa": {"acc,none": 0.7, "acc_stderr,none": 0.02},
            "mmlu_anatomy": {"acc,none": 0.6, "acc_stderr,none": 0.04},
        }
    }

    metrics = benchmark_metrics(results, ["medqa_4options", "pubmedqa"])

    assert metrics == {
        "benchmark_medqa_4options_acc": 0.5,
        "benchmark_medqa_4options_acc_stderr": 0.01,
        "benchmark_pubmedqa_acc": 0.7,
        "benchmark_pubmedqa_acc_stderr": 0.02,
    }


def test_benchmark_metrics_names_a_missing_task() -> None:
    with pytest.raises(KeyError, match="medmcqa"):
        benchmark_metrics({"results": {}}, ["medmcqa"])


def test_harness_model_args_turn_thinking_off_and_keep_the_training_context() -> None:
    args = harness_model_args("checkpoints/best", 2048)

    assert args == {
        "pretrained": "checkpoints/best",
        "dtype": "bfloat16",
        "enable_thinking": False,
        "max_length": 2048,
    }


def test_perplexity_metrics_keeps_the_word_perplexity_per_task() -> None:
    results = {"results": {"wikitext": {"word_perplexity,none": 12.5, "byte_perplexity,none": 1.6}}}

    assert perplexity_metrics(results, ["wikitext"]) == {"benchmark_wikitext_word_perplexity": 12.5}


def test_perplexity_metrics_names_a_missing_task() -> None:
    with pytest.raises(KeyError, match="wikitext"):
        perplexity_metrics({"results": {}}, ["wikitext"])


def test_pubmedqa_test_keeps_the_official_test_questions() -> None:
    dataset = Dataset.from_dict({"pubid": [21645374, 1, 2]})

    assert official_test(dataset)["pubid"] == [21645374]


def test_pubmedqa_test_prompt_matches_the_harness_task() -> None:
    doc = {"context": {"contexts": ["First.", "Second."]}, "question": "Does X help?"}

    assert doc_to_text(doc) == "Abstract: First.\nSecond.\nQuestion: Does X help?\nAnswer:"


def test_benchmarks_config_lists_the_medical_tasks() -> None:
    benchmarks = load_multitask_config().benchmarks

    assert "medqa_4options" in benchmarks.tasks
    assert "mmlu_professional_medicine" in benchmarks.tasks
    assert benchmarks.num_fewshot == 0


def test_benchmarks_config_lists_general_tasks_to_measure_forgetting() -> None:
    benchmarks = load_multitask_config().benchmarks

    assert benchmarks.general_tasks == ["mmlu", "arc_challenge", "hellaswag"]


def test_benchmark_tasks_runs_the_mmlu_medical_subjects_inside_mmlu() -> None:
    # The harness refuses a subject listed next to its group; the group's results hold it
    config = load_multitask_config()

    tasks = benchmark_tasks(config.benchmarks)

    assert tasks == ["medqa_4options", "medmcqa", "pubmedqa_test", "mmlu", "arc_challenge", "hellaswag", "wikitext"]


def test_benchmark_tasks_keeps_a_subject_whose_group_is_not_run() -> None:
    benchmarks = BenchmarksConfig(
        tasks=["medmcqa", "mmlu_anatomy"],
        general_tasks=["arc_challenge"],
        perplexity_tasks=["wikitext"],
        num_fewshot=0,
        batch_size=16,
    )

    assert benchmark_tasks(benchmarks) == ["medmcqa", "mmlu_anatomy", "arc_challenge", "wikitext"]


def test_benchmarks_config_measures_forgetting_as_wikitext_perplexity() -> None:
    assert load_multitask_config().benchmarks.perplexity_tasks == ["wikitext"]
