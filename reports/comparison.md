# Model comparison

Test metrics. Accuracies and G-Eval scores: mean over seeds with a 95% bootstrap confidence interval over the test questions; other metrics: mean ± std over seeds. `zero-shot` is the model without fine-tuning; runs are named `<method>_<start>`, where `original` starts from the Hub model and `dapt` from the PubMed-pretrained one.

| Model | Run | MedText loss | PubMedQA acc | PubMedQA F1 | MedMCQA acc | MedMCQA F1 | MedQA acc | MedQA F1 | G-Eval correctness | G-Eval completeness | G-Eval coherence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | zero-shot | 2.002 ± 0.000 | 0.340 [0.298, 0.382] | 0.171 ± 0.000 | 0.309 [0.293, 0.325] | 0.305 ± 0.000 | 0.300 [0.275, 0.327] | 0.298 ± 0.000 | - | - | - |
| qwen3-1.7b | lora_original | 1.569 ± 0.000 | 0.754 [0.716, 0.792] | 0.541 ± 0.000 | 0.400 [0.384, 0.417] | 0.391 ± 0.000 | 0.410 [0.382, 0.438] | 0.407 ± 0.000 | 0.614 [0.558, 0.670] | 0.535 [0.487, 0.582] | 0.937 [0.904, 0.965] |
|  | chance / majority class |  | 0.552 |  | 0.250 |  | 0.250 |  |  |  |  |

## Fine-tuning vs zero-shot

Paired bootstrap over the same test questions.

| Model | Run | Task | Δ accuracy vs zero-shot [95% CI] | p-value |
|---|---|---|---|---|
| qwen3-1.7b | lora_original | PubMedQA | +0.414 [+0.362, +0.466] | <0.001 |
| qwen3-1.7b | lora_original | MedMCQA | +0.091 [+0.073, +0.109] | <0.001 |
| qwen3-1.7b | lora_original | MedQA | +0.110 [+0.078, +0.141] | <0.001 |

## Standard benchmarks

lm-evaluation-harness, 0-shot, multiple choice with the options shown, in the chat format: accuracy ± standard error. Comparable with published results.

| Model | Run | medqa_4options | medmcqa | pubmedqa_test | mmlu_anatomy | mmlu_clinical_knowledge | mmlu_college_biology | mmlu_college_medicine | mmlu_medical_genetics | mmlu_professional_medicine | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | zero-shot | 0.358 ± 0.013 | 0.384 ± 0.008 | 0.338 ± 0.021 | 0.430 ± 0.043 | 0.385 ± 0.030 | 0.493 ± 0.042 | 0.364 ± 0.037 | 0.400 ± 0.049 | 0.309 ± 0.028 | 0.384 |
| qwen3-1.7b | lora_original | 0.452 ± 0.014 | 0.448 ± 0.008 | 0.740 ± 0.020 | 0.511 ± 0.043 | 0.513 ± 0.031 | 0.625 ± 0.040 | 0.555 ± 0.038 | 0.590 ± 0.049 | 0.533 ± 0.030 | 0.552 |
