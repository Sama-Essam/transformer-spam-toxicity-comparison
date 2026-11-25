"""
Unified Evaluation & Comparison Pipeline (Member 2 Deliverable).

Evaluates BOTH:
- Model A: Custom Transformer from Scratch (Member 1)
- Model B: Fine-tuned Pretrained Transformer (DistilBERT - Member 2)

Evaluates strictly on the exact same held-out test split: data/test.csv (559 samples).

Computes & Exports:
- Accuracy, Macro-F1, Macro Precision & Recall
- Per-class Precision, Recall, F1-Score & Support for Ham and Spam
- Confusion Matrices (saved as JSON, CSV, and high-res PNG plots)
- Average Single-Sample Latency (ms)
- Trainable Parameter Count & Training Duration
- Side-by-side structured comparison:
  * artifacts/model_a_metrics.json
  * artifacts/model_b_metrics.json
  * results/comparison_metrics.json
  * results/comparison_table.csv
  * results/metrics_comparison.png
  * results/confusion_matrices_comparison.png
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.scratch.inference import ScratchModelPredictor
from models.pretrained.inference import PretrainedModelPredictor


def load_test_data(test_csv: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Loads text, label strings, and label IDs from test.csv.
    """
    texts, true_labels, true_ids = [], [], []
    with open(test_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            true_labels.append(row["label"])
            true_ids.append(int(row["label_id"]))
    return texts, true_labels, true_ids


def evaluate_single_model(
    predictor,
    texts: List[str],
    true_ids: List[int],
    training_time_seconds: float = 0.0,
    hardware_name: str = "CPU"
) -> Dict[str, Any]:
    """
    Runs sample-by-sample inference for exact per-sample latency measurement and metric calculation.
    """
    pred_labels = []
    latencies = []

    for text in texts:
        res = predictor.predict(text)
        pred_labels.append(res["label"])
        latencies.append(res["latency_ms"])

    pred_ids = [0 if p == "Ham" else 1 for p in pred_labels]
    avg_latency = float(np.mean(latencies))

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
        "num_test_samples": len(texts),
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
        "trainable_parameters": getattr(predictor, "trainable_params", 0),
        "training_time_seconds": round(training_time_seconds, 2),
        "hardware": hardware_name
    }
    return metrics


def plot_confusion_matrix(cm: List[List[int]], title: str, save_path: str, acc: float, f1: float):
    """
    Plots and saves a high-resolution confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title(f"{title}\nAccuracy: {acc*100:.2f}% | Macro-F1: {f1:.4f}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot: {save_path}")


def plot_side_by_side_confusion_matrices(
    cm_a: List[List[int]],
    cm_b: List[List[int]],
    save_path: str,
    acc_a: float, f1_a: float,
    acc_b: float, f1_b: float
):
    """
    Generates a dual-panel figure with Model A and Model B confusion matrices side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Model A Heatmap
    sns.heatmap(
        cm_a,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[0].set_title(f"Model A (Scratch Transformer)\nAcc: {acc_a*100:.2f}% | Macro-F1: {f1_a:.4f}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Label", fontsize=11, fontweight="bold")

    # Model B Heatmap
    sns.heatmap(
        cm_b,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=axes[1],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[1].set_title(f"Model B (Pretrained DistilBERT)\nAcc: {acc_b*100:.2f}% | Macro-F1: {f1_b:.4f}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Label", fontsize=11, fontweight="bold")

    plt.suptitle("Side-by-Side Confusion Matrix Comparison (Held-Out Test Set: 559 samples)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side confusion matrix comparison: {save_path}")


def generate_comparison_charts(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], save_path: str):
    """
    Generates a comprehensive 4-panel visual benchmarking chart comparing:
    1. Primary Classification Metrics (Accuracy, Macro-F1, Precision, Recall)
    2. Per-Class F1 Scores (Ham vs Spam)
    3. Inference Latency (ms per sample)
    4. Trainable Parameter Count (Footprint)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = ["#4C72B0", "#55A868"]  # Blue (Model A), Green (Model B)

    # 1. Primary Classification Metrics
    metric_names = ["Accuracy", "Macro-F1", "Macro Precision", "Macro Recall"]
    scores_a = [metrics_a["accuracy"], metrics_a["macro_f1"], metrics_a["macro_precision"], metrics_a["macro_recall"]]
    scores_b = [metrics_b["accuracy"], metrics_b["macro_f1"], metrics_b["macro_precision"], metrics_b["macro_recall"]]

    x = np.arange(len(metric_names))
    width = 0.35

    ax1 = axes[0, 0]
    rects1 = ax1.bar(x - width/2, [s * 100 for s in scores_a], width, label="Model A (Scratch)", color=palette[0])
    rects2 = ax1.bar(x + width/2, [s * 100 for s in scores_b], width, label="Model B (DistilBERT)", color=palette[1])
    ax1.set_ylabel("Score (%)", fontweight="bold")
    ax1.set_title("Overall Test Classification Metrics", fontweight="bold", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, fontweight="bold")
    ax1.set_ylim(85, 101)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Per-Class F1 Score Comparison
    ax2 = axes[0, 1]
    class_names = ["Ham (Majority)", "Spam (Minority)"]
    f1_class_a = [metrics_a["class_metrics"]["Ham"]["f1_score"] * 100, metrics_a["class_metrics"]["Spam"]["f1_score"] * 100]
    f1_class_b = [metrics_b["class_metrics"]["Ham"]["f1_score"] * 100, metrics_b["class_metrics"]["Spam"]["f1_score"] * 100]

    x2 = np.arange(len(class_names))
    rects1_c = ax2.bar(x2 - width/2, f1_class_a, width, label="Model A (Scratch)", color=palette[0])
    rects2_c = ax2.bar(x2 + width/2, f1_class_b, width, label="Model B (DistilBERT)", color=palette[1])
    ax2.set_ylabel("F1 Score (%)", fontweight="bold")
    ax2.set_title("Per-Class F1 Score (Ham vs. Spam)", fontweight="bold", fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(class_names, fontweight="bold")
    ax2.set_ylim(85, 101)
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1_c:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2_c:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Average Single-Sample Latency (ms)
    ax3 = axes[1, 0]
    lat_models = ["Model A (Scratch)", "Model B (DistilBERT)"]
    latencies = [metrics_a["average_inference_latency_ms"], metrics_b["average_inference_latency_ms"]]
    bars3 = ax3.bar(lat_models, latencies, color=palette, width=0.45)
    ax3.set_ylabel("Latency (ms / sample)", fontweight="bold")
    ax3.set_title("Single-Sample Inference Latency", fontweight="bold", fontsize=12)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars3:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f} ms", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 4. Trainable Parameter Count (Footprint)
    ax4 = axes[1, 1]
    params = [metrics_a["trainable_parameters"] / 1e6, metrics_b["trainable_parameters"] / 1e6]
    bars4 = ax4.bar(lat_models, params, color=palette, width=0.45)
    ax4.set_ylabel("Parameters (Millions)", fontweight="bold")
    ax4.set_title("Model Parameter Footprint (Trainable)", fontweight="bold", fontsize=12)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars4:
        h = bar.get_height()
        ax4.annotate(f"{h:.2f} M", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Model A (Scratch) vs. Model B (Pretrained DistilBERT) Test Benchmark", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side metric comparison chart: {save_path}")


def export_comparison_table(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], csv_path: str):
    """
    Exports clean, human-readable CSV comparison table.
    """
    rows = [
        {"Metric": "Model Architecture", "Model A (From Scratch)": "Custom Transformer (Encoder-Only)", "Model B (Pretrained)": "DistilBERT (distilbert-base-uncased)"},
        {"Metric": "Test Accuracy", "Model A (From Scratch)": f"{metrics_a['accuracy']*100:.2f}%", "Model B (Pretrained)": f"{metrics_b['accuracy']*100:.2f}%"},
        {"Metric": "Macro-F1 Score", "Model A (From Scratch)": f"{metrics_a['macro_f1']:.4f}", "Model B (Pretrained)": f"{metrics_b['macro_f1']:.4f}"},
        {"Metric": "Macro Precision", "Model A (From Scratch)": f"{metrics_a['macro_precision']*100:.2f}%", "Model B (Pretrained)": f"{metrics_b['macro_precision']*100:.2f}%"},
        {"Metric": "Macro Recall", "Model A (From Scratch)": f"{metrics_a['macro_recall']*100:.2f}%", "Model B (Pretrained)": f"{metrics_b['macro_recall']*100:.2f}%"},
        {"Metric": "Ham F1-Score", "Model A (From Scratch)": f"{metrics_a['class_metrics']['Ham']['f1_score']:.4f}", "Model B (Pretrained)": f"{metrics_b['class_metrics']['Ham']['f1_score']:.4f}"},
        {"Metric": "Spam F1-Score", "Model A (From Scratch)": f"{metrics_a['class_metrics']['Spam']['f1_score']:.4f}", "Model B (Pretrained)": f"{metrics_b['class_metrics']['Spam']['f1_score']:.4f}"},
        {"Metric": "Ham Precision / Recall", "Model A (From Scratch)": f"{metrics_a['class_metrics']['Ham']['precision']*100:.2f}% / {metrics_a['class_metrics']['Ham']['recall']*100:.2f}%", "Model B (Pretrained)": f"{metrics_b['class_metrics']['Ham']['precision']*100:.2f}% / {metrics_b['class_metrics']['Ham']['recall']*100:.2f}%"},
        {"Metric": "Spam Precision / Recall", "Model A (From Scratch)": f"{metrics_a['class_metrics']['Spam']['precision']*100:.2f}% / {metrics_a['class_metrics']['Spam']['recall']*100:.2f}%", "Model B (Pretrained)": f"{metrics_b['class_metrics']['Spam']['precision']*100:.2f}% / {metrics_b['class_metrics']['Spam']['recall']*100:.2f}%"},
        {"Metric": "Confusion Matrix [[TN, FP], [FN, TP]]", "Model A (From Scratch)": str(metrics_a['confusion_matrix']), "Model B (Pretrained)": str(metrics_b['confusion_matrix'])},
        {"Metric": "Avg. Single-Sample Latency", "Model A (From Scratch)": f"{metrics_a['average_inference_latency_ms']:.2f} ms", "Model B (Pretrained)": f"{metrics_b['average_inference_latency_ms']:.2f} ms"},
        {"Metric": "Trainable Parameters", "Model A (From Scratch)": f"{metrics_a['trainable_parameters']:,}", "Model B (Pretrained)": f"{metrics_b['trainable_parameters']:,}"},
        {"Metric": "Training Time", "Model A (From Scratch)": f"{metrics_a['training_time_seconds']:.2f} s", "Model B (Pretrained)": f"{metrics_b['training_time_seconds']:.2f} s ({metrics_b['training_time_seconds']/60.0:.2f} min)"},
        {"Metric": "Evaluation Hardware", "Model A (From Scratch)": metrics_a.get("hardware", "CPU"), "Model B (Pretrained)": metrics_b.get("hardware", "CPU")},
    ]

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")
    return df


def run_unified_evaluation(
    test_csv: str = "data/test.csv",
    artifacts_dir: str = "artifacts",
    results_dir: str = "results",
    model_a_path: str = "artifacts/model_a_scratch.pt",
    model_a_tok: str = "artifacts/scratch_tokenizer.json",
    model_b_path: str = "artifacts/model_b_pretrained",
    model_a_training_time: float = None,
    model_b_training_time: float = None,
    hardware: str = "CPU"
):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("🚀 UNIFIED TEST SET EVALUATION: MODEL A vs. MODEL B")
    print("=" * 65)
    print(f"Held-Out Test Set: {test_csv}")

    # Load test instances
    texts, true_labels, true_ids = load_test_data(test_csv)
    n_samples = len(texts)
    print(f"Evaluating {n_samples} held-out test samples across both models...")

    # Load prior training info if available
    if model_a_training_time is None:
        a_metrics_file = os.path.join(artifacts_dir, "model_a_metrics.json")
        if os.path.exists(a_metrics_file):
            try:
                with open(a_metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    model_a_training_time = data.get("training_time_seconds", 7.93)
            except Exception:
                model_a_training_time = 7.93
        else:
            model_a_training_time = 7.93

    if model_b_training_time is None:
        b_info_file = os.path.join(artifacts_dir, "model_b_training_info.json")
        if os.path.exists(b_info_file):
            try:
                with open(b_info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    model_b_training_time = data.get("fine_tuning_time_seconds", 2111.49)
            except Exception:
                model_b_training_time = 2111.49
        else:
            model_b_training_time = 2111.49

    # 1. Evaluate Model A
    print("\n--- Evaluating Model A (Custom Transformer from Scratch) ---")
    pred_a = ScratchModelPredictor(model_path=model_a_path, tokenizer_path=model_a_tok)
    metrics_a = evaluate_single_model(
        pred_a, texts, true_ids,
        training_time_seconds=model_a_training_time,
        hardware_name=hardware
    )
    print(f"Model A Results -> Accuracy: {metrics_a['accuracy']*100:.2f}% | Macro-F1: {metrics_a['macro_f1']:.4f} | Avg Latency: {metrics_a['average_inference_latency_ms']:.2f} ms")

    # Save Model A metrics & CM
    a_json_path = os.path.join(artifacts_dir, "model_a_metrics.json")
    with open(a_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_a, f, indent=2)

    a_cm_plot = os.path.join(artifacts_dir, "model_a_confusion_matrix.png")
    plot_confusion_matrix(
        metrics_a["confusion_matrix"],
        "Model A Confusion Matrix (Scratch Transformer)",
        a_cm_plot,
        metrics_a["accuracy"],
        metrics_a["macro_f1"]
    )

    # 2. Evaluate Model B
    print("\n--- Evaluating Model B (Fine-Tuned Pretrained DistilBERT) ---")
    pred_b = PretrainedModelPredictor(model_path_or_name=model_b_path)
    metrics_b = evaluate_single_model(
        pred_b, texts, true_ids,
        training_time_seconds=model_b_training_time,
        hardware_name=hardware
    )
    print(f"Model B Results -> Accuracy: {metrics_b['accuracy']*100:.2f}% | Macro-F1: {metrics_b['macro_f1']:.4f} | Avg Latency: {metrics_b['average_inference_latency_ms']:.2f} ms")

    # Save Model B metrics & CM
    b_json_path = os.path.join(artifacts_dir, "model_b_metrics.json")
    with open(b_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_b, f, indent=2)

    b_cm_plot = os.path.join(artifacts_dir, "model_b_confusion_matrix.png")
    plot_confusion_matrix(
        metrics_b["confusion_matrix"],
        "Model B Confusion Matrix (Pretrained DistilBERT)",
        b_cm_plot,
        metrics_b["accuracy"],
        metrics_b["macro_f1"]
    )

    # 3. Side-by-Side Visualizations & Comparison Files
    print("\n--- Generating Side-by-Side Comparative Deliverables ---")

    # Dual confusion matrices
    dual_cm_path = os.path.join(results_dir, "confusion_matrices_comparison.png")
    plot_side_by_side_confusion_matrices(
        metrics_a["confusion_matrix"],
        metrics_b["confusion_matrix"],
        dual_cm_path,
        metrics_a["accuracy"], metrics_a["macro_f1"],
        metrics_b["accuracy"], metrics_b["macro_f1"]
    )

    # 4-panel metrics chart
    comp_chart_path = os.path.join(results_dir, "metrics_comparison.png")
    generate_comparison_charts(metrics_a, metrics_b, comp_chart_path)

    # JSON Comparison
    comparison_summary = {
        "evaluation_task": "Spam / Toxicity Binary Text Classification",
        "evaluation_split": "data/test.csv",
        "num_test_samples": n_samples,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_a": metrics_a,
        "model_b": metrics_b,
        "delta": {
            "accuracy_diff": round(metrics_b["accuracy"] - metrics_a["accuracy"], 4),
            "macro_f1_diff": round(metrics_b["macro_f1"] - metrics_a["macro_f1"], 4),
            "latency_ratio": round(metrics_b["average_inference_latency_ms"] / max(metrics_a["average_inference_latency_ms"], 1e-6), 2),
            "parameter_ratio": round(metrics_b["trainable_parameters"] / max(metrics_a["trainable_parameters"], 1), 2)
        }
    }
    comp_json_path = os.path.join(results_dir, "comparison_metrics.json")
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"Saved side-by-side comparison JSON to {comp_json_path}")

    # CSV Table
    comp_table_path = os.path.join(results_dir, "comparison_table.csv")
    df = export_comparison_table(metrics_a, metrics_b, comp_table_path)

    print("\n" + "=" * 65)
    print("📊 FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 65)
    print(df.to_string(index=False))

    return comparison_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Compare Model A vs Model B on Test Set")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to held-out test CSV")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Path to artifacts directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--hardware", type=str, default="CPU", help="Hardware device name")
    parser.add_argument("--model_a_time", type=float, default=None, help="Model A training time (seconds)")
    parser.add_argument("--model_b_time", type=float, default=None, help="Model B training time (seconds)")
    args = parser.parse_args()

    run_unified_evaluation(
        test_csv=args.test_csv,
        artifacts_dir=args.artifacts_dir,
        results_dir=args.results_dir,
        model_a_training_time=args.model_a_time,
        model_b_training_time=args.model_b_time,
        hardware=args.hardware
    )
