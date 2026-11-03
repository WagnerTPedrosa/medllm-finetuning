# medllm-finetuning

Fine-tuning a base LLM (Qwen3-1.7B-Base) for medical question answering on
MedText, PubMedQA, MedMCQA and MedQA, with LoRA, optional continued
pretraining on PubMed papers, and a report with confidence intervals. The
first experiment (GPT-2) is archived at git tag `v1-gpt2`.

## Layout

```
configs/                 Hydra configuration
  config.yaml            the whole pipeline
  model/<model>.yaml     values per base model (learning rates, batch sizes, ...)
pipelines/               DVC pipelines
  data/dvc.yaml          download -> preprocess -> split -> filter_length@model
  training/dvc.yaml      evaluate_zero_shot -> finetune_lora_original -> judge_geval -> report
src/medllm/
  config.py              typed configuration (frozen dataclasses)
  modeling.py            loading models, LoRA, GPU memory
  tracking.py            MLflow runs and Model Registry
  data/                  download, preprocessing, splits, mixture, multitask (CLI), corpus
  training/              finetune, sweep, pretrain
  evaluation/            loss, choices, tasks, zero_shot, generation, geval
  reporting/             report, plots
tests/                   mirrors src/medllm/
notebooks/               exploration
data/                    raw/ and interim/ (rebuilt), processed/ (per model), benchmarks/
results/<model>/         one JSON summary per training stage
reports/comparison.md    the final comparison table
```

## Running

```bash
uv sync
uv run dvc repro pipelines/data/dvc.yaml:filter_length@qwen3-1.7b
uv run dvc repro pipelines/training/dvc.yaml:report
```

Name the last stage you need: `dvc repro` of a whole file also checks the
frozen PubMed pretraining stages and fails. `uv run dvc status
pipelines/training/dvc.yaml` shows what would run and why. Each stage can also
be run on its own, for example
`uv run python -m medllm.training.finetune --base original model=qwen3-1.7b`.

## Tracking

Runs, metrics and models are tracked in MLflow (`mlflow.db`):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Tests

```bash
uv run pytest
```
