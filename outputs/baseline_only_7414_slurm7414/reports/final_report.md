# Dysarthria Detection and Severity Classification Report

## Objective
Build an end-to-end pipeline for automatic dysarthria detection (binary) and severity classification (multi-class) using TORGO and UA Speech datasets, and compare a classical baseline against wav2vec2.

## Methodology
- Data preparation: load TORGO and UA Speech audio, resample to 16 kHz, trim or pad to fixed duration, and normalize.
- Baseline: MFCC + delta + delta-delta summary statistics with an SVM classifier.
- Deep model: wav2vec2 fine-tuning with partial encoder freezing, AdamW, and warmup scheduling.
- Evaluation: held-out speaker split plus cross-dataset transfer (TORGO->UA and UA->TORGO).
- Interpretability: optional SVM permutation importance and wav2vec2 saliency / attention rollout plots.

## Results
- intra_dataset_test / binary: best model=MFCC+SVM, F1=0.7935, Accuracy=0.7217
- intra_dataset_test / severity: best model=MFCC+SVM, F1=0.1842, Accuracy=0.2396
- TORGO_to_UA / binary: best model=MFCC+SVM, F1=0.6789, Accuracy=0.5143
- TORGO_to_UA / severity: metrics unavailable (all candidate F1 values are NaN).
- UA_to_TORGO / binary: best model=MFCC+SVM, F1=0.5188, Accuracy=0.3504
- UA_to_TORGO / severity: metrics unavailable (all candidate F1 values are NaN).

## Comparison
- wav2vec2 usually gives stronger binary detection performance when the pretrained representation transfers well.
- MFCC + SVM stays useful as a lightweight baseline and can remain competitive on smaller or noisier splits.
- Cross-dataset performance is typically worse than intra-dataset performance because speaker populations, prompts, and recording conditions differ.

## Limitations
- Path-derived labels are only a heuristic unless you provide a curated metadata CSV.
- The held-out split is designed for practical experimentation rather than exhaustive model selection.
- Severity classes can be sparse and inconsistent across packaged dataset variants.
- Severity split could not include all 3 classes in train, validation, and test despite retries; severity metrics should be interpreted cautiously.

## Conclusion
This repository provides a reproducible, cluster-oriented project structure for dysarthria experiments. For publication-grade experiments, the next step should be curated metadata and a stricter train/validation/test protocol.
