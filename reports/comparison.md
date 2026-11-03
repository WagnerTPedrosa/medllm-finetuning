# Base model comparison

Test metrics of the multitask fine-tuned models, mean ± std over seeds. `original` starts from the Hub model, `dapt` from the PubMed-pretrained one.

| Model | Base | MedText loss | PubMedQA acc | PubMedQA F1 | MedMCQA acc | MedMCQA F1 | G-Eval correctness | G-Eval completeness | G-Eval coherence |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | original | 2.034 ± 0.009 | 0.638 ± 0.017 | 0.447 ± 0.009 | 0.304 ± 0.028 | 0.221 ± 0.021 | 0.111 ± 0.023 | 0.111 ± 0.024 | 0.489 ± 0.009 |
| gpt2 | dapt | 1.976 ± 0.011 | 0.651 ± 0.023 | 0.455 ± 0.012 | 0.298 ± 0.032 | 0.210 ± 0.010 | 0.104 ± 0.015 | 0.097 ± 0.012 | 0.494 ± 0.048 |
