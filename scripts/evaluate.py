"""
Standardized Test Evaluation Script for Model A (Member 1).
Evaluates best Model A checkpoint strictly on held-out data/test.csv.
Computes:
- Accuracy & Macro-F1
- Per-class and Macro Precision / Recall
- Confusion Matrix (counts & plot saved to artifacts/model_a_confusion_matrix.png)
- Average Inference Latency (ms per sample)
- Trainable Parameter Count
- Outputs standardized artifacts/model_a_metrics.json
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import Dict, Any
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.scratch.inference import ScratchModelPredictor


def evaluate_test_set(
    test_csv: str = "data/test.csv",
    model_path: str = "artifacts/model_a_scratch.pt",
    tokenizer_path: str = "artifacts/scratch_tokenizer.json",
    artifacts_dir: str = "artifacts",
    training_time_seconds: float = 0.0,
    hardware_name: str = "CPU"
) -> Dict[str, Any]:
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Initialize predictor
    predictor = ScratchModelPredictor(model_path=model_path, tokenizer_path=tokenizer_path)

    # 2. Read test set
    texts, true_labels, true_ids = [], [], []
    with open(test_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            true_labels.append(row["label"])
            true_ids.append(int(row["label_id"]))

    n_samples = len(texts)
    print(f"Evaluating Model A on {n_samples} held-out test samples...")

    # 3. Run inference sample-by-sample for exact latency measurement
    pred_labels = []
    latencies = []

    for text in texts:
        res = predictor.predict(text)
        pred_labels.append(res["label"])
        latencies.append(res["latency_ms"])

    pred_ids = [0 if p == "Ham" else 1 for p in pred_labels]
    avg_latency = float(np.mean(latencies))

    # 4. Calculate metrics
    acc = float(accuracy_score(true_ids, pred_ids))
    macro_f1 = float(f1_score(true_ids, pred_ids, average="macro", zero_division=0))
    macro_precision = float(precision_score(true_ids, pred_ids, average="macro", zero_division=0))
    macro_recall = float(recall_score(true_ids, pred_ids, average="macro", zero_division=0))

    cm = confusion_matrix(true_ids, pred_ids).tolist()

    class_report = classification_report(
        true_ids, pred_ids, target_names=["Ham", "Spam"], output_dict=True, zero_division=0
    )

    metrics = {
        "model_name": predictor.model_name,
        "task": "Spam / Toxicity Binary Text Classification",
        "evaluation_split": "test.csv",
        "num_test_samples": n_samples,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "class_metrics": {
            "Ham": {
                "precision": round(class_report["Ham"]["precision"], 4),
                "recall": round(class_report["Ham"]["recall"], 4),
                "f1_score": round(class_report["Ham"]["f1-score"], 4),
                "support": class_report["Ham"]["support"]
            },
            "Spam": {
                "precision": round(class_report["Spam"]["precision"], 4),
                "recall": round(class_report["Spam"]["recall"], 4),
                "f1_score": round(class_report["Spam"]["f1-score"], 4),
                "support": class_report["Spam"]["support"]
            }
        },
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["Ham (0)", "Spam (1)"],
        "average_inference_latency_ms": round(avg_latency, 2),
        "trainable_parameters": predictor.trainable_params,
        "training_time_seconds": round(training_time_seconds, 2),
        "hardware": hardware_name
    }

    # 5. Save metrics JSON
    metrics_path = os.path.join(artifacts_dir, "model_a_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    # 6. Save Confusion Matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"]
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Model A Confusion Matrix\n(Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f})")
        plt.tight_layout()

        cm_path = os.path.join(artifacts_dir, "model_a_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot to {cm_path}")
    except Exception as e:
        print(f"Plotting confusion matrix skipped or encountered warning: {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model A on Test Set")
    parser.add_argument("--training_time", type=float, default=0.0, help="Total training time in seconds")
    parser.add_argument("--hardware", type=str, default="CPU", help="Hardware device name used")
    args = parser.parse_args()

    results = evaluate_test_set(
        training_time_seconds=args.training_time,
        hardware_name=args.hardware
    )
    print("\nFinal Test Metrics Summary:")
    print(json.dumps(results, indent=2))
