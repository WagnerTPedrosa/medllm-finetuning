"""Loss curves for single runs and hyperparameter searches."""

from operator import attrgetter, itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from matplotlib.figure import Figure

# Training loss in the Trainer's log history
TRAIN_LOSS_KEY = "loss"
LOSS_CURVES_ARTIFACT = "plots/loss_curves.png"


def loss_curves_figure(log_history: list[dict[str, float]], eval_key: str) -> Figure:
    """Training and validation loss against the optimizer step, with the best evaluation marked.

    Where the validation loss rises while the training loss keeps falling,
    the model is overfitting.

    Args:
        log_history: `trainer.state.log_history` (every entry has a `step`).
        eval_key: Validation loss to plot, e.g. `eval_combined_loss`.

    Returns:
        The figure (close it with `plt.close` when done).

    Raises:
        ValueError: If no entry has `eval_key`.
    """
    train_logs: list[dict[str, float]] = [log for log in log_history if TRAIN_LOSS_KEY in log]
    eval_logs: list[dict[str, float]] = [log for log in log_history if eval_key in log]
    if not eval_logs:
        raise ValueError(f"No {eval_key} in the log history")

    best: dict[str, float] = min(eval_logs, key=itemgetter(eval_key))
    figure, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([log["step"] for log in train_logs], [log[TRAIN_LOSS_KEY] for log in train_logs], label="train loss", alpha=0.8)
    ax.plot([log["step"] for log in eval_logs], [log[eval_key] for log in eval_logs], label=eval_key, marker="o")
    ax.axvline(best["step"], color="grey", linestyle="--", label=f"best: {best[eval_key]:.4f} at step {best['step']}")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()
    return figure


def log_loss_curves(log_history: list[dict[str, float]], eval_key: str) -> None:
    """Save the loss curves of a training run as `plots/loss_curves.png` in the active MLflow run."""
    figure: Figure = loss_curves_figure(log_history, eval_key)
    mlflow.log_figure(figure, LOSS_CURVES_ARTIFACT)
    plt.close(figure)


def plot_run_metrics(
    run_id: str,
    metric_names: list[str],
    title: str = "Continued pretraining",
    save_path: str | Path = "experiments/pretrain_curves.png",
) -> None:
    """Plot the step history of MLflow metrics of one run, one panel per metric.

    Args:
        run_id: MLflow run id.
        metric_names: Metrics to plot, e.g. `["loss", "eval_perplexity"]`.
        title: Figure title.
        save_path: Where to save the PNG.

    Raises:
        ValueError: If metric_names is empty.
    """
    if not metric_names:
        raise ValueError("metric_names must not be empty")

    client = mlflow.MlflowClient()
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 4), squeeze=False)

    for ax, name in zip(axes[0], metric_names):
        history = sorted(client.get_metric_history(run_id, name), key=attrgetter("step"))
        ax.plot([point.step for point in history], [point.value for point in history], marker=".")
        ax.set_title(name)
        ax.set_xlabel("Optimizer step")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)
