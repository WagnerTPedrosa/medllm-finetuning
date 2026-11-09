from pathlib import Path

import mlflow
import pytest

from medllm.config import MultitaskConfig
from medllm.reporting.plots import log_loss_curves, loss_curves_figure
from medllm.tracking import configure_tracking

LOG_HISTORY: list[dict[str, float]] = [
    {"loss": 2.0, "step": 10},
    {"loss": 1.8, "step": 20},
    {"eval_combined_loss": 1.6, "eval_combined_runtime": 3.0, "step": 20},
    {"loss": 1.5, "step": 30},
    {"loss": 1.4, "step": 40},
    {"eval_combined_loss": 1.7, "step": 40},
    {"train_loss": 1.7, "step": 40},
]


def test_loss_curves_figure_plots_training_and_validation_loss_by_step() -> None:
    figure = loss_curves_figure(LOG_HISTORY, "eval_combined_loss")

    lines = {line.get_label(): line for line in figure.axes[0].get_lines()}
    assert list(lines["train loss"].get_xdata()) == [10, 20, 30, 40]
    assert list(lines["train loss"].get_ydata()) == [2.0, 1.8, 1.5, 1.4]
    assert list(lines["eval_combined_loss"].get_xdata()) == [20, 40]
    assert list(lines["eval_combined_loss"].get_ydata()) == [1.6, 1.7]


def test_loss_curves_figure_marks_the_best_evaluation() -> None:
    figure = loss_curves_figure(LOG_HISTORY, "eval_combined_loss")

    best = [line for line in figure.axes[0].get_lines() if line.get_label().startswith("best")]
    assert len(best) == 1
    assert list(best[0].get_xdata()) == [20, 20]


def test_loss_curves_figure_needs_an_evaluation() -> None:
    with pytest.raises(ValueError, match="eval_loss"):
        loss_curves_figure([{"loss": 2.0, "step": 10}], "eval_loss")


def test_log_loss_curves_saves_the_figure_in_the_active_run(
    config: MultitaskConfig, tracking_store: Path
) -> None:
    configure_tracking(config)

    with mlflow.start_run() as run:
        log_loss_curves(LOG_HISTORY, "eval_combined_loss")

    artifacts = [item.path for item in mlflow.MlflowClient().list_artifacts(run.info.run_id, "plots")]
    assert artifacts == ["plots/loss_curves.png"]
