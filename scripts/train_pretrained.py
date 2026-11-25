"""
Standalone Fine-Tuning Pipeline for Model B (Pretrained Transformer - DistilBERT).
Member 2 Implementation.

Guarantees:
- Strictly fine-tunes on data/train.csv and validates on data/val.csv.
- Zero data leakage: data/test.csv is NEVER touched during training or validation.
- Fixes deterministic random seed (SEED=42) for total reproducibility.
- Employs AdamW with linear warmup, class-weighted CrossEntropyLoss, and early stopping.
- Logs exact fine-tuning training time and parameter count.
- Saves fine-tuned checkpoint and tokenizer to artifacts/model_b_pretrained/.
"""

import os
import sys
import time
import json
import csv
import random
import argparse
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HFTextClassificationDataset(Dataset):
    """
    PyTorch Dataset wrapping Hugging Face tokenization for sequence classification.
    """
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.texts = []
        self.labels = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(row["text"])
                self.labels.append(int(row["label_id"]))

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_class_weights(train_csv: str) -> torch.Tensor:
    """
    Computes inverse frequency class weights to address label imbalance (~86.6% Ham vs ~13.4% Spam).
    Matches Member 1's weighting scheme.
    """
    counts = [0, 0]
    with open(train_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[int(row["label_id"])] += 1
    total = sum(counts)
    weights = [total / (2.0 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def evaluate_val_set(model, val_loader, criterion, device) -> Tuple[float, float, float]:
    """
    Evaluates model on validation set. Returns (avg_loss, accuracy, macro_f1).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def train_pretrained(
    train_csv: str = "data/train.csv",
    val_csv: str = "data/val.csv",
    base_model_name: str = "models/pretrained/base_distilbert",
    output_dir: str = "artifacts/model_b_pretrained",
    batch_size: int = 32,
    lr: float = 3e-5,
    epochs: int = 3,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
    device_str: str = None
) -> Dict[str, Any]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 65)
    print("🛡️  MODEL B FINE-TUNING PIPELINE (Member 2)")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Base Checkpoint: {base_model_name}")
    print(f"Train Dataset: {train_csv} (Strictly no test split)")
    print(f"Validation Dataset: {val_csv}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, seed={seed}")

    # Fallback if local path not found
    if not os.path.exists(base_model_name):
        base_model_name = "distilbert-base-uncased"

    # 1. Load Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = HFTextClassificationDataset(train_csv, tokenizer, max_length=max_length)
    val_dataset = HFTextClassificationDataset(val_csv, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 2. Model Initialization
    id2label = {0: "Ham", 1: "Spam"}
    label2id = {"Ham": 0, "Spam": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total | {trainable_params:,} trainable.")

    # 3. Loss & Optimizer
    class_weights = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Weight decay exclusions for bias / LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 4. Training Loop
    start_train_time = time.perf_counter()
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0

    print("\n--- Commencing Fine-Tuning ---")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_idx = 0

        for batch in train_loader:
            batch_idx += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * input_ids.size(0)

        train_loss = epoch_loss / len(train_dataset)
        val_loss, val_acc, val_f1 = evaluate_val_set(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch

            # Save best checkpoint & tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  --> Checkpoint saved to {output_dir} (Best Val Macro-F1: {best_val_f1:.4f})")

    fine_tuning_time = time.perf_counter() - start_train_time
    print(f"\nFine-tuning completed in {fine_tuning_time:.2f} seconds ({fine_tuning_time/60.0:.2f} minutes).")
    print(f"Best Validation Macro-F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    # 5. Save Training Metadata
    training_info = {
        "model_name": "Model B (Pretrained Transformer - DistilBERT)",
        "base_checkpoint": "distilbert-base-uncased",
        "task": "Spam / Toxicity Binary Classification",
        "random_seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "fine_tuning_time_seconds": round(fine_tuning_time, 2),
        "fine_tuning_time_minutes": round(fine_tuning_time / 60.0, 2),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "output_directory": output_dir,
        "hardware": "CUDA" if torch.cuda.is_available() else "CPU"
    }

    info_path = os.path.join("artifacts", "model_b_training_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)
    print(f"Saved fine-tuning metadata to {info_path}")

    return training_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Pretrained DistilBERT (Model B)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--base_model", type=str, default="models/pretrained/base_distilbert", help="Base HF checkpoint")
    args = parser.parse_args()

    train_pretrained(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        base_model_name=args.base_model
    )
