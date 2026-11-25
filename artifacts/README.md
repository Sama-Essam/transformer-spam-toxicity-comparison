# Artifacts Directory

This directory stores serialized weights, tokenizers, evaluation metrics, and plots for both Model A and Model B.

### Files Schema:
- `model_a_scratch.pt`: PyTorch weights checkpoint for Model A (Member 1).
- `scratch_tokenizer.json`: Serialized custom vocabulary and tokenizer for Model A (Member 1).
- `model_a_metrics.json`: Standardized JSON test metrics for Model A.
- `model_a_confusion_matrix.png`: Confusion matrix visualization for Model A on `data/test.csv`.
- `model_b_pretrained/`: Serialized fine-tuned weights (`model.safetensors`), configuration (`config.json`), and tokenizer (`tokenizer.json`, `vocab.txt`) for Model B (Member 2).
- `model_b_training_info.json`: Training metadata, hyperparameter configuration, and fine-tuning duration for Model B.
- `model_b_metrics.json`: Standardized JSON test metrics for Model B on `data/test.csv`.
- `model_b_confusion_matrix.png`: Confusion matrix visualization for Model B on `data/test.csv`.

### Unified Results (in `results/`):
- `results/comparison_metrics.json`: Hierarchical JSON comparing both models side-by-side on `data/test.csv`.
- `results/comparison_table.csv`: Human-readable comparative metrics table.
- `results/metrics_comparison.png`: 4-panel comparative bar chart (metrics, per-class F1, latency, parameters).
- `results/confusion_matrices_comparison.png`: Dual-panel side-by-side confusion matrix figure.
