# Error Analysis Artifact Manifest

This commit includes aggregate metrics and test-set prediction CSVs for the successful baseline, wav2vec2, WavLM base-plus, and WavLM large runs.

Prediction CSVs include the test metadata columns plus:

- `y_true`: ground-truth label used for scoring
- `y_pred`: model prediction
- `is_correct`: per-row correctness flag

Large model checkpoints under `models/` are intentionally excluded from Git.

## Included Runs

| Model | Task | Run directory | Prediction CSV |
| --- | --- | --- | --- |
| MFCC+SVM baseline | Binary | `outputs/baseline_only_7414_slurm7414` | `tables/baseline_binary_test_predictions.csv` |
| MFCC+SVM baseline | Severity | `outputs/baseline_only_7414_slurm7414` | `tables/baseline_severity_test_predictions.csv` |
| wav2vec2 | Binary | `outputs/wav2vec2_binary_a100_7227_slurm7227` | `tables/wav2vec_binary_test_predictions.csv` |
| wav2vec2 | Severity | `outputs/wav2vec2_severity_a100_7396_slurm7396` | `tables/wav2vec_severity_test_predictions.csv` |
| WavLM base-plus | Binary | `outputs/wavlm_base_plus_binary_a100_7228_slurm7228` | `tables/wav2vec_binary_test_predictions.csv` |
| WavLM base-plus | Severity | `outputs/wavlm_base_plus_severity_a100_7397_slurm7397` | `tables/wav2vec_severity_test_predictions.csv` |
| WavLM large | Binary | `outputs/wavlm_large_binary_a100_7615_slurm7615` | `tables/wav2vec_binary_test_predictions.csv` |
| WavLM large | Severity | `outputs/wavlm_large_severity_a100_7616_slurm7616` | `tables/wav2vec_severity_test_predictions.csv` |

## Test Prediction Row Counts

| Run | Rows | Accuracy from `is_correct` |
| --- | ---: | ---: |
| `baseline_only_7414_slurm7414` binary | 29,574 | 0.721749 |
| `baseline_only_7414_slurm7414` severity | 15,300 | 0.239608 |
| `wav2vec2_binary_a100_7227_slurm7227` | 29,574 | 0.902583 |
| `wav2vec2_severity_a100_7396_slurm7396` | 15,300 | 0.100784 |
| `wavlm_base_plus_binary_a100_7228_slurm7228` | 29,574 | 0.810374 |
| `wavlm_base_plus_severity_a100_7397_slurm7397` | 15,300 | 0.097124 |
| `wavlm_large_binary_a100_7615_slurm7615` | 29,574 | 0.857950 |
| `wavlm_large_severity_a100_7616_slurm7616` | 15,300 | 0.041373 |

Each included run directory also contains its Hydra config snapshots, environment summary, final report, classification report, model comparison table, and training log.
