

from medllm.evaluation.tasks import correctness


def test_correctness_marks_each_prediction() -> None:
    assert correctness(["A", "B", "C"], ["A", "C", "C"]) == [1, 0, 1]
