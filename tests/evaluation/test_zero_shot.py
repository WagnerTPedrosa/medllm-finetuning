import pytest

from medllm.evaluation.zero_shot import baseline_stage


def test_baseline_stage_keeps_the_original_name_for_the_hub_model() -> None:
    assert baseline_stage("original") == "baseline"


def test_baseline_stage_names_the_pretrained_model_baseline() -> None:
    assert baseline_stage("dapt") == "baseline_dapt"


def test_baseline_stage_rejects_an_unknown_base() -> None:
    with pytest.raises(ValueError, match="base"):
        baseline_stage("other")
