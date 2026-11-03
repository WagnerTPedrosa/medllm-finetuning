import asyncio
from pathlib import Path

import pytest
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

from src.config import GEvalConfig, MultitaskConfig, load_multitask_config
from src.geval import (
    GEVAL_METRICS,
    build_geval_metrics,
    build_judge,
    build_test_cases,
    score_test_cases,
    seed_model_dir,
    summarize_geval,
)

MULTITASK_CONFIG_PATH = Path("configs/gpt2_multitask.yaml").resolve()


def load() -> MultitaskConfig:
    return load_multitask_config(MULTITASK_CONFIG_PATH)


class FakeJudge(DeepEvalBaseLLM):
    """Judge that gives every answer the same score out of 10."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.prompts: list[str] = []

    def load_model(self) -> None:
        return None

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> BaseModel:
        self.prompts.append(prompt)
        return schema(reason="Fake reason.", score=self.score)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return "fake-judge"


def test_load_multitask_config_reads_geval_section() -> None:
    geval: GEvalConfig = load().geval

    assert geval.base_url.startswith("http://localhost")
    assert geval.max_concurrent > 0
    assert geval.max_new_tokens > 0


def test_build_judge_points_at_the_configured_server() -> None:
    geval = load().geval

    judge = build_judge(geval)

    assert judge.name == geval.judge_model
    assert judge.base_url == geval.base_url.rstrip("/")
    assert judge.temperature == 0.0


def test_build_geval_metrics_has_one_metric_per_criterion() -> None:
    metrics = build_geval_metrics(FakeJudge(score=5))

    assert [metric.name for metric in metrics] == list(GEVAL_METRICS)
    correctness = metrics[0]
    assert SingleTurnParams.EXPECTED_OUTPUT in correctness.evaluation_params


def test_build_test_cases_pairs_prompts_answers_and_references() -> None:
    cases = build_test_cases(["Q1", "Q2"], ["a1", "a2"], ["r1", "r2"])

    assert [(case.input, case.actual_output, case.expected_output) for case in cases] == [
        ("Q1", "a1", "r1"),
        ("Q2", "a2", "r2"),
    ]


def test_build_test_cases_rejects_lists_of_different_lengths() -> None:
    with pytest.raises(ValueError):
        build_test_cases(["Q1", "Q2"], ["a1"], ["r1", "r2"])


def test_score_test_cases_returns_a_normalized_score_and_reason_per_metric() -> None:
    cases = [LLMTestCase(input="Q", actual_output="A", expected_output="R")] * 3

    rows = asyncio.run(score_test_cases(cases, FakeJudge(score=7), max_concurrent=2))

    assert len(rows) == 3
    for name in GEVAL_METRICS:
        assert rows[0][name] == pytest.approx(0.7)
        assert rows[0][f"{name}_reason"] == "Fake reason."


def test_summarize_geval_averages_scores_and_ignores_reasons() -> None:
    rows = [
        {"correctness": 0.2, "correctness_reason": "x", "coherence": 1.0},
        {"correctness": 0.6, "correctness_reason": "y", "coherence": 0.0},
    ]

    summary = summarize_geval(rows, ["correctness", "coherence"], prefix="test_medtext_geval")

    assert summary == pytest.approx({"test_medtext_geval_correctness": 0.4, "test_medtext_geval_coherence": 0.5})


def test_seed_model_dir_follows_the_training_layout() -> None:
    config = load()

    model_dir = seed_model_dir(config, base_model="models:/gpt2-pubmed-dapt@candidate", seed="43")

    assert model_dir == Path(config.finetune.output_dir) / "gpt2-pubmed-dapt@candidate-seed43" / "best"
