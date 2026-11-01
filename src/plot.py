from pathlib import Path
import matplotlib.pyplot as plt


def plot_losses(
    train_losses,
    val_losses,
    baseline_val_loss=None,
    title="GPT-2 Fine-Tuning - MedText",
    save_path="experiments/loss_curve.png",
):
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Training loss
    ax.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss",
    )

    # Validation loss + optional baseline
    val_epochs = epochs
    val_values = val_losses

    if baseline_val_loss is not None:
        val_epochs = [0] + epochs
        val_values = [baseline_val_loss] + val_losses

    ax.plot(
        val_epochs,
        val_values,
        marker="o",
        label="Validation Loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.set_xticks(val_epochs)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)


def plot_grid_search(
    results,
    baseline_val_loss=None,
    save_path="experiments/grid_search_losses.png",
):
    if not results:
        raise ValueError("results must contain at least one grid search trial")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    colors = plt.get_cmap("tab10")
    best_trial = min(results, key=lambda result: result["best_val_loss"])
    labels = []
    handles = []

    for index, result in enumerate(results):
        epochs = range(1, len(result["train_losses"]) + 1)
        label = (
            f'batch={result["batch_size"]}, '
            f'lr={result["learning_rate"]:g}'
        )
        color = colors(index % 10)
        axes[0].plot(epochs, result["train_losses"], marker="o", color=color)
        line, = axes[1].plot(
            range(1, len(result["val_losses"]) + 1),
            result["val_losses"],
            marker="o",
            color=color,
            label=label,
        )
        handles.append(line)
        labels.append(label)

    if baseline_val_loss is not None:
        baseline = axes[1].axhline(
            baseline_val_loss, color="gray", linestyle="--", label="Baseline"
        )
        handles.append(baseline)
        labels.append("Baseline")

    for ax, title in zip(axes[:2], ("Training loss", "Validation loss")):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(True, alpha=0.3)

    point_colors = [
        "tab:orange" if result is best_trial else "tab:blue"
        for result in results
    ]
    axes[2].scatter(
        [result["best_val_loss"] for result in results],
        range(len(results)),
        color=point_colors,
        s=80,
    )
    axes[2].set_yticks(
        range(len(results)),
        [
            f'batch={result["batch_size"]}\nlr={result["learning_rate"]:g}'
            for result in results
        ],
    )
    axes[2].set_title("Best validation loss by parameters")
    axes[2].set_xlabel("Cross-entropy loss")
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.3)

    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.13, 1, 1))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)
