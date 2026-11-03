import asyncio
import stat
from dataclasses import replace
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
    JudgeServer,
    build_serve_command,
    build_test_cases,
    mean_scores_per_example,
    score_test_cases,
    server_is_up,
    summarize_geval,
)



def load() -> MultitaskConfig:
    return load_multitask_config()


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


def test_build_serve_command_runs_vllm_on_the_configured_port() -> None:
    geval = replace(load().geval, base_url="http://localhost:8123/v1")

    command = build_serve_command(geval)

    assert command[:3] == [str(Path(geval.vllm_bin) / "vllm"), "serve", geval.judge_model]
    assert "--port=8123" in command
    assert command[-len(geval.serve_args) :] == geval.serve_args


def test_server_is_up_is_false_when_nothing_listens() -> None:
    assert server_is_up("http://localhost:9/v1") is False


def fake_vllm(directory: Path, script: str) -> GEvalConfig:
    """GEval config whose `vllm` is a shell script, on a port where nothing listens."""
    executable = directory / "vllm"
    executable.write_text("#!/bin/sh\n" + script + "\n")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    return replace(
        load().geval,
        base_url="http://localhost:9/v1",
        vllm_bin=str(directory),
        server_log=str(directory / "vllm.log"),
        startup_timeout=2,
    )


def test_judge_server_reports_a_server_that_exits(tmp_path: Path) -> None:
    geval = fake_vllm(tmp_path, "echo 'bad flag'; exit 1")

    with pytest.raises(RuntimeError, match="vllm.log"):
        with JudgeServer(geval):
            pass

    assert "bad flag" in (tmp_path / "vllm.log").read_text()


def test_judge_server_stops_a_server_that_never_gets_ready(tmp_path: Path) -> None:
    geval = fake_vllm(tmp_path, "sleep 600")
    server = JudgeServer(geval)

    with pytest.raises(TimeoutError):
        server.__enter__()

    assert server.process is not None and server.process.poll() is not None


def test_mean_scores_per_example_averages_the_seeds() -> None:
    seed_rows = [
        [{"correctness": 0.2, "correctness_reason": "x"}, {"correctness": 1.0, "correctness_reason": "y"}],
        [{"correctness": 0.4, "correctness_reason": "z"}, {"correctness": 0.0, "correctness_reason": "w"}],
    ]

    scores = mean_scores_per_example(seed_rows, ["correctness"])

    assert scores == {"correctness": pytest.approx([0.3, 0.5])}
