# Model comparison

Test metrics. Accuracies and G-Eval scores: mean over seeds with a 95% bootstrap confidence interval over the test questions; other metrics: mean ± std over seeds. `zero-shot` is the model without fine-tuning and `zero-shot_dapt` the PubMed-pretrained one; runs are named `<method>_<start>`, where `original` starts from the Hub model and `dapt` from the PubMed-pretrained one.

| Model | Run | MedText loss | PubMedQA acc | PubMedQA F1 | MedMCQA acc | MedMCQA F1 | MedQA acc | MedQA F1 | G-Eval correctness | G-Eval completeness | G-Eval coherence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | zero-shot | 2.002 ± 0.000 | 0.340 [0.298, 0.382] | 0.171 ± 0.000 | 0.309 [0.293, 0.325] | 0.305 ± 0.000 | 0.300 [0.275, 0.327] | 0.298 ± 0.000 | - | - | - |
| qwen3-1.7b | zero-shot_dapt | 1.967 ± 0.000 | 0.342 [0.300, 0.384] | 0.174 ± 0.000 | 0.296 [0.281, 0.312] | 0.293 ± 0.000 | 0.305 [0.279, 0.332] | 0.303 ± 0.000 | - | - | - |
| qwen3-1.7b | lora_dapt | 1.605 ± 0.000 | 0.752 [0.714, 0.788] | 0.550 ± 0.000 | 0.392 [0.376, 0.409] | 0.384 ± 0.000 | 0.391 [0.363, 0.419] | 0.388 ± 0.000 | 0.564 [0.507, 0.620] | 0.520 [0.470, 0.570] | 0.951 [0.923, 0.975] |
| qwen3-1.7b | lora_original | 1.569 ± 0.000 | 0.754 [0.716, 0.792] | 0.541 ± 0.000 | 0.400 [0.384, 0.417] | 0.391 ± 0.000 | 0.410 [0.382, 0.438] | 0.407 ± 0.000 | 0.611 [0.556, 0.668] | 0.532 [0.485, 0.579] | 0.937 [0.901, 0.965] |
|  | chance / majority class |  | 0.552 |  | 0.250 |  | 0.250 |  |  |  |  |

## Fine-tuning vs zero-shot

Paired bootstrap over the same test questions.

| Model | Run | Task | Δ accuracy vs zero-shot [95% CI] | p-value |
|---|---|---|---|---|
| qwen3-1.7b | zero-shot_dapt | PubMedQA | +0.002 [-0.004, +0.008] | 0.783 |
| qwen3-1.7b | zero-shot_dapt | MedMCQA | -0.013 [-0.024, -0.002] | 0.030 |
| qwen3-1.7b | zero-shot_dapt | MedQA | +0.005 [-0.014, +0.025] | 0.631 |
| qwen3-1.7b | lora_dapt | PubMedQA | +0.412 [+0.362, +0.462] | <0.001 |
| qwen3-1.7b | lora_dapt | MedMCQA | +0.083 [+0.065, +0.101] | <0.001 |
| qwen3-1.7b | lora_dapt | MedQA | +0.091 [+0.058, +0.124] | <0.001 |
| qwen3-1.7b | lora_original | PubMedQA | +0.414 [+0.362, +0.466] | <0.001 |
| qwen3-1.7b | lora_original | MedMCQA | +0.091 [+0.073, +0.109] | <0.001 |
| qwen3-1.7b | lora_original | MedQA | +0.110 [+0.078, +0.141] | <0.001 |

## With vs without PubMed pretraining

Each fine-tuning of the pretrained model against the same fine-tuning of the Hub model, paired over the same test questions.

| Model | Run | Task | Δ accuracy vs without pretraining [95% CI] | p-value |
|---|---|---|---|---|
| qwen3-1.7b | lora_dapt | PubMedQA | -0.002 [-0.028, +0.024] | 0.922 |
| qwen3-1.7b | lora_dapt | MedMCQA | -0.008 [-0.020, +0.003] | 0.156 |
| qwen3-1.7b | lora_dapt | MedQA | -0.019 [-0.039, +0.002] | 0.079 |

## Standard benchmarks

lm-evaluation-harness, 0-shot, multiple choice with the options shown, in the chat format: accuracy ± standard error. Comparable with published results. `zero-shot_dapt` is the PubMed-pretrained model before fine-tuning.

| Model | Run | medqa_4options | medmcqa | pubmedqa_test | mmlu_anatomy | mmlu_clinical_knowledge | mmlu_college_biology | mmlu_college_medicine | mmlu_medical_genetics | mmlu_professional_medicine | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | zero-shot | 0.358 ± 0.013 | 0.384 ± 0.008 | 0.338 ± 0.021 | 0.430 ± 0.043 | 0.385 ± 0.030 | 0.493 ± 0.042 | 0.364 ± 0.037 | 0.400 ± 0.049 | 0.309 ± 0.028 | 0.385 |
| qwen3-1.7b | zero-shot_dapt | 0.284 ± 0.013 | 0.327 ± 0.007 | 0.338 ± 0.021 | 0.215 ± 0.035 | 0.219 ± 0.025 | 0.271 ± 0.037 | 0.208 ± 0.031 | 0.300 ± 0.046 | 0.191 ± 0.024 | 0.261 |
| qwen3-1.7b | lora_original | 0.452 ± 0.014 | 0.448 ± 0.008 | 0.740 ± 0.020 | 0.511 ± 0.043 | 0.513 ± 0.031 | 0.625 ± 0.040 | 0.555 ± 0.038 | 0.590 ± 0.049 | 0.533 ± 0.030 | 0.552 |
| qwen3-1.7b | lora_dapt | 0.433 ± 0.014 | 0.455 ± 0.008 | 0.752 ± 0.019 | 0.563 ± 0.043 | 0.551 ± 0.031 | 0.667 ± 0.039 | 0.514 ± 0.038 | 0.600 ± 0.049 | 0.485 ± 0.030 | 0.558 |

## General knowledge

The same, on general tasks: a drop from `zero-shot` means the medical training made the model forget general knowledge.

| Model | Run | mmlu | arc_challenge | hellaswag | Average |
|---|---|---|---|---|---|
| qwen3-1.7b | zero-shot | 0.416 ± 0.004 | 0.346 ± 0.014 | 0.453 ± 0.005 | 0.405 |
| qwen3-1.7b | zero-shot_dapt | 0.245 ± 0.004 | 0.304 ± 0.013 | 0.422 ± 0.005 | 0.324 |
| qwen3-1.7b | lora_original | 0.500 ± 0.004 | 0.462 ± 0.015 | 0.460 ± 0.005 | 0.474 |
| qwen3-1.7b | lora_dapt | 0.513 ± 0.004 | 0.428 ± 0.014 | 0.442 ± 0.005 | 0.461 |

Word perplexity on WikiText (raw Wikipedia text, windows of the training context; lower is better): a rise from `zero-shot` means the model lost some of its plain language modelling.

| Model | Run | wikitext |
|---|---|---|
| qwen3-1.7b | zero-shot | 13.77 |
| qwen3-1.7b | zero-shot_dapt | 15.28 |
| qwen3-1.7b | lora_original | 21.57 |
| qwen3-1.7b | lora_dapt | 23.30 |
