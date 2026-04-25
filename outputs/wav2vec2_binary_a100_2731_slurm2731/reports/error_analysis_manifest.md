# Error Analysis Manifest

This run directory contains the non-model artifacts needed to inspect the `wav2vec2_binary_a100_2731_slurm2731` results later.

Useful files:

- `config.resolved.yaml`: exact resolved Hydra config for the run
- `environment.json`: runtime environment summary
- `metadata/test_binary.csv`: held-out binary split before unreadable-audio filtering
- `reports/final_report.md`: high-level run summary
- `reports/wav2vec_binary_classification_report.txt`: aggregate binary metrics
- `tables/model_comparison.csv`: compact summary table
- `training_logs.json`: per-epoch training history

Important note:

- The Slurm run removed `1` unreadable row from the binary test split before evaluation, so the evaluated binary test set size was `29,574` even though `metadata/test_binary.csv` contains `29,575` rows.

Not included here:

- `models/wav2vec2_binary_best.pt` was not pushed because the file is about `361 MB`, which exceeds GitHub's normal file size limit without Git LFS.
