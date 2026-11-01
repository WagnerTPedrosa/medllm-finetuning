import torch
import json
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from src.evaluation import evaluate


def train(
    model,
    epochs,
    optimizer,
    device,
    train_loader,
    val_loader,
    best_global_loss=float("inf"),
):
    best_trial_loss = float("inf")
    best_epoch = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            num_tokens = (batch["labels"][:, 1:] != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        train_loss = total_loss / total_tokens
        val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Best validation loss within this trial
        if val_loss < best_trial_loss:
            best_trial_loss = val_loss
            best_epoch = epoch + 1

        # Best validation loss across all trials
        if val_loss < best_global_loss:
            best_global_loss = val_loss

            output_dir = Path("checkpoints/gpt2-medtext-best")
            output_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(output_dir)

    metrics = {
        "best_val_loss": best_trial_loss,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    return metrics, best_global_loss


def run_grid_search(
    train_tokenized,
    val_tokenized,
    collate_fn,
    device,
):
    param_grid = {
        "learning_rate": [1e-5, 2e-5, 5e-5],
        "batch_size": [2, 8],
    }
    
    results = []
    best_global_loss = float("inf")
    combinations = list(ParameterGrid(param_grid))

    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, params in enumerate(combinations, start=1):
        print(f"\nTrial {idx}/{len(combinations)}")
        print(f"Parameters: {params}")

        # Reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        generator = torch.Generator().manual_seed(42)

        # DataLoaders for this batch size
        train_loader = DataLoader(
            train_tokenized,
            batch_size=params["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            generator=generator,
        )

        val_loader = DataLoader(
            val_tokenized,
            batch_size=params["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Fresh pretrained GPT-2 for every trial
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)

        model.config.use_cache = False

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=0.01,
        )

        metrics, best_global_loss = train(
            model=model,
            epochs=5,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            best_global_loss=best_global_loss,
        )

        results.append(
            {
                "trial": idx,
                **params,
                **metrics,
            }
        )

        # Save results after each experiment
        with open(output_dir / "hpo_results.json", "w") as f:
            json.dump(results, f, indent=4)

        del model, optimizer, train_loader, val_loader

    # Find the best hyperparameter combination
    best = min(results, key=lambda result: result["best_val_loss"])

    print("\nBest hyperparameters:")
    print(best)

    return best, results
