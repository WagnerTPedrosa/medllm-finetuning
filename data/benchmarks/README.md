# Benchmark splits

- `pubmedqa_test_ground_truth.json`: the official PubMedQA test set (500 PMIDs of
  PQA-L with their gold decision), copied from
  https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json.
  The remaining 500 PQA-L questions are split into 450 train / 50 validation
  (stratified by decision) by `src/multitask.py`, as in BLURB.
