"""Loss curves for single runs and hyperparameter searches."""

from operator import attrgetter
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow


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
