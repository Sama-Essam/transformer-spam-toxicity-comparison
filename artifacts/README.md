# Artifacts Directory

This directory stores serialized weights, tokenizers, evaluation metrics, and plots.

### Files Schema:
- `model_a_scratch.pt`: PyTorch weights checkpoint for Model A (Member 1).
- `scratch_tokenizer.json`: Serialized custom vocabulary and tokenizer (Member 1).
- `model_a_metrics.json`: Standardized JSON test metrics for Model A:
  ```json
  {
    "model_name": "Model A (Custom Transformer - From Scratch)",
    "accuracy": 0.0,
    "macro_f1": 0.0,
    "macro_precision": 0.0,
    "macro_recall": 0.0,
    "class_metrics": {},
    "confusion_matrix": [],
    "latency_ms_per_sample": 0.0,
    "trainable_parameters": 0,
    "training_time_seconds": 0.0,
    "hardware": "NVIDIA T4"
  }
  ```
- `model_a_confusion_matrix.png`: Confusion matrix visualization for Model A.
- `model_b_metrics.json`: Standardized JSON test metrics for Model B (Member 2).
