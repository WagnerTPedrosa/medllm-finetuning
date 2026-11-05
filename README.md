# medllm-finetuning

## Goal

Build a small language model that answers medical questions well, and later make it usable through a dashboard with a chat interface.

The model is [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base), fine-tuned with LoRA on several medical question-answering datasets at once (four for now; more may be added). The main question this project asks is whether **pretraining on medical papers first** (to learn the language of the domain) makes the fine-tuned model better. The same fine-tuning is therefore run twice:

| Run | Starting model |
|---|---|
| `lora_original` | Qwen3-1.7B-Base as published |
| `lora_dapt` | Qwen3-1.7B-Base after further pretraining on PubMed papers (domain-adaptive pretraining) |

Both are compared with the base model before any fine-tuning (zero-shot).

## Datasets

| Dataset | What it is | Answer |
|---|---|---|
| [MedText](https://huggingface.co/datasets/BI55/MedText) | 1,412 clinical scenarios and questions, with detailed answers | free text |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | research questions about PubMed abstracts: 1,000 labelled by experts (PQA-L) and 211k generated automatically (PQA-A) | yes / no / maybe, with the abstract's conclusion |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | 194k questions from Indian medical entrance exams (AIIMS, NEET PG), many with an explanation | one of 4 options |
| [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | 11k clinical vignettes from the US medical licensing exam (USMLE) | one of 4 options |

Every dataset is downloaded at a fixed version, so the data can be rebuilt exactly.

## Data preparation

All four datasets are turned into the same format: a question and the answer the model should write.

```
Question:
A 60-year-old man with a history of angina is now experiencing more frequent
and severe angina attacks that are triggered by less exertion than before.
What could this signify?

Answer:
This could indicate that the patient's stable angina is progressing towards
unstable angina. ...
```

**MedMCQA and MedQA become open questions.** The goal is a model that answers questions, not one that picks a letter, so the options are removed and the answer is the text of the correct option. For MedMCQA the explanation is added after it:

```
Question:
Axonal transport is:

Answer:
Antegrade and retrograde

Explanation: Fast anterograde (400 mm/day) transport occurs by kinesin
molecular motor and retrograde transport (200 mm/day) occurs by dynein
molecular motor.
```

- "Which of the following..." is rewritten as "What...".
- Questions that only make sense with the options are removed ("All of the following are true except", "Which is false", "Both", "None of the above"): 27% of MedMCQA and 7% of MedQA.
- MedMCQA explanations are cleaned: the restated answer ("Ans. is 'c' i.e.") and textbook references ("Ref: Harrison 19th ed") are removed, and explanations with almost nothing left are dropped.

**PubMedQA** keeps its abstract as context, and the answer is the decision followed by the abstract's conclusion:

```
Question:
Do mitochondria play a role in remodelling lace plant leaves during
programmed cell death?

Answer (yes, no or maybe):
yes. Results depicted mitochondrial dynamics in vivo as PCD progresses ...
```

PQA-A is 93% "yes", so each epoch uses a balanced sample of 15,000 "yes" and 15,000 "no" questions. Articles in the PubMedQA test set are removed from it.

**Splits.** The official test sets are kept for the final results. MedMCQA's official test set has no answers, so its official validation set is used as the test set instead. A small validation set is held out from each training set to choose the best epoch. Examples longer than the model's context (2,048
tokens) are removed, never cut.

| Dataset | Train | Validation | Test |
|---|---|---|---|
| MedText | 1,129 | 141 | 142 |
| PubMedQA | 450 (+ 211,269 PQA-A) | 50 | 500 |
| MedMCQA | 133,821 | 150 | 3,246 |
| MedQA | 9,362 | 150 | 1,176 |

**Training mixture.** The datasets differ a lot in size, so each epoch mixes them: MedText and PubMedQA (PQA-L) 5 times, the balanced PQA-A sample, 60,000 new MedMCQA questions and MedQA twice, about 117,000 examples per epoch.

## PubMed corpus (for `lora_dapt`)

1,000,000 open-access articles from PubMed Central (mostly full text), sampled across medical topics. Before pretraining:

- articles used in PubMedQA are removed, so the model is not tested on papers it has read;
- administrative sections (funding, conflicts of interest, author contributions, ethics statements) and citation markers like `[1-3]` are removed;
- 0.1% of the articles (about 1,000) are kept apart to measure how well the model predicts medical text.

This gives about 6 billion tokens of training text.

## Training

- **LoRA** on all attention and MLP layers (rank 16): the base model is frozen and small adapter weights are trained, then merged into the model.
- The model learns to write the answer only (the question is not part of the loss).
- Up to 5 epochs; after each one the model is scored on the validation sets, and training stops when it no longer improves.

## Evaluation

| Dataset | How it is scored |
|---|---|
| MedText | an LLM judge (Qwen3.8-27B) rates the generated answers for correctness, completeness and coherence (G-Eval) |
| PubMedQA | accuracy: is yes / no / maybe the most likely answer? |
| MedMCQA, MedQA | accuracy: of the 4 option texts, is the correct one the most likely answer to the open question? |

Every score has a 95% confidence interval, and every fine-tuned model is compared with zero-shot on the same questions to tell whether the difference is real. The model never sees the options, so the MedMCQA and MedQA scores are not comparable with published multiple-choice results.

## Results

To be added: `reports/comparison.md` is written at the end of each pipeline run.

## Reproducing

```bash
uv sync
uv run dvc repro pipelines/data/dvc.yaml:filter_length@qwen3-1.7b
uv run dvc repro pipelines/training/dvc.yaml:report
uv run pytest
```

The data and training steps are DVC pipelines in `pipelines/`, the configuration is in `configs/` (Hydra), and the code is the `medllm` package in `src/medllm/`. Runs and models are tracked with MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`). The first experiment, with GPT-2, is archived at git tag `v1-gpt2`.
