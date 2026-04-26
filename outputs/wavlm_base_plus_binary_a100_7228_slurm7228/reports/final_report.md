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
- intra_dataset_test / binary: best model=wav2vec2, F1=0.8608, Accuracy=0.8104
- No result available for severity / intra_dataset_test.
- No result available for binary / TORGO_to_UA.
- No result available for severity / TORGO_to_UA.
- No result available for binary / UA_to_TORGO.
- No result available for severity / UA_to_TORGO.

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
