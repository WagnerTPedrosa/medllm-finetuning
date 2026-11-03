# medllm-finetuning

Training saves the best fine-tuned model to `checkpoints/gpt2-medtext-best/`.
Checkpoints are generated locally and are not stored in this Git repository.

## MLflow experiment tracking

Install the project dependencies with `uv sync`. The notebook's existing
`run_grid_search(...)` and `run_optuna_search(...)` calls then create one MLflow
run per trial. Each run stores the effective training parameters, the complete
YAML configuration, training and validation losses by epoch. The prompt-completion
training and validation splits appear under Datasets, and the fine-tuned model
and tokenizer appear under Logged Models. Optuna trials
stopped by pruning retain their current weights and have a `status=pruned` tag;
completed trials save the best validation checkpoint. The results JSON files
include each trial's `mlflow_run_id` and `mlflow_model_id` (when available).

Tracking uses the SQLite URI and experiment name in
`configs/gpt2_medtext.yaml` (currently `mlflow.db` and
`gpt2-medtext-fine-tuned`). Start the UI from the project root:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the URL printed by MLflow (usually `http://127.0.0.1:5000`). To use a
remote tracking server, set `MLFLOW_TRACKING_URI` before running the notebook;
set `MLFLOW_EXPERIMENT_NAME` to change the experiment name. Model artifacts can
take substantial disk space because every trial gets its own copy. A saved
trial model can be loaded with
`mlflow.transformers.load_model(f"models:/{model_id}")`.
If the named experiment was deleted in MLflow, new runs go into a name ending
in `-rerun` (or the next available suffix) without restoring the deleted one.

Runs created before dataset and model entities were added can be upgraded with
`uv run python -m src.repair_mlflow_runs`. The command is safe to repeat and
processes only completed runs in the configured experiment.
