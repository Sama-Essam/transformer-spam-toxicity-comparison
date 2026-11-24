"""
Standalone Training Pipeline for Model A (Custom Transformer from Scratch).
Features:
- Fits SimpleTokenizer strictly on data/train.csv (zero data leakage)
- Class-weighted Cross-Entropy to handle class imbalance
- AdamW optimizer with warmup + cosine decay
- Early stopping based on validation Macro-F1
- Saves checkpoint with model kwargs to artifacts/model_a_scratch.pt
"""

import os
import sys
import time
import math
import json
import csv
import argparse
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.scratch.transformer import TransformerClassifier
from models.scratch.tokenizer import SimpleTokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: SimpleTokenizer):
        self.samples = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                label_id = int(row["label_id"])
                input_ids, attention_mask = tokenizer.encode(text)
                self.samples.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(label_id, dtype=torch.long)
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def compute_class_weights(train_csv: str) -> torch.Tensor:
    counts = [0, 0]
    with open(train_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[int(row["label_id"])] += 1
    total = sum(counts)
    # Balanced weights: total / (num_classes * count)
    weights = [total / (2.0 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for input_ids, mask, labels in loader:
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for input_ids, mask, labels in loader:
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask=mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def train_model(
    train_csv: str = "data/train.csv",
    val_csv: str = "data/val.csv",
    artifacts_dir: str = "artifacts",
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    max_length: int = 128,
    max_vocab_size: int = 8000,
    batch_size: int = 64,
    lr: float = 3e-4,
    epochs: int = 15,
    patience: int = 4,
    device_str: str = None
):
    os.makedirs(artifacts_dir, exist_ok=True)
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {device}")

    # 1. Fit Tokenizer strictly on train.csv
    tokenizer_path = os.path.join(artifacts_dir, "scratch_tokenizer.json")
    tokenizer = SimpleTokenizer(max_length=max_length)
    tokenizer.fit_from_csv(train_csv, max_vocab_size=max_vocab_size)
    tokenizer.save(tokenizer_path)

    # 2. Datasets & Loaders
    train_dataset = TextClassificationDataset(train_csv, tokenizer)
    val_dataset = TextClassificationDataset(val_csv, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model Architecture
    model_kwargs = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "max_len": max_length,
        "num_classes": 2,
        "dropout": 0.1,
        "pad_idx": tokenizer.pad_id,
        "pooling": "mean"
    }
    model = TransformerClassifier(**model_kwargs).to(device)
    print(f"Model A instantiated: {model.count_parameters():,} trainable parameters.")

    # 4. Loss & Optimizer
    class_weights = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # 5. Training Loop with Early Stopping on Validation Macro-F1
    best_val_f1 = -1.0
    patience_counter = 0
    checkpoint_path = os.path.join(artifacts_dir, "model_a_scratch.pt")

    start_train_time = time.perf_counter()
    print("\n--- Starting Model A Training ---")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc * 100:.2f}% | "
              f"Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "model_kwargs": model_kwargs,
                "vocab_size": tokenizer.vocab_size
            }, checkpoint_path)
            print(f"  --> Checkpoint saved (Best Val Macro-F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    total_training_time = time.perf_counter() - start_train_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds.")
    print(f"Best Checkpoint: {checkpoint_path} with Val Macro-F1: {best_val_f1:.4f}")

    return {
        "best_val_f1": best_val_f1,
        "training_time_seconds": round(total_training_time, 2),
        "checkpoint_path": checkpoint_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model A Custom Transformer from Scratch")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
