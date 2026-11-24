# Transformer: From Scratch vs. Pretrained Model
### Undergraduate NLP Course Project — Spam & Toxicity Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io/)

A modular comparative NLP study investigating the performance, computational efficiency, parameter footprint, and inference latency between:
1. **Model A:** A custom Encoder-only Transformer classifier implemented **completely from scratch** using raw PyTorch primitives (no `nn.Transformer` or pretrained weights).
2. **Model B:** A fine-tuned modern Pretrained Transformer (e.g., DistilBERT / BERT / RoBERTa via Hugging Face).

---

## 👥 3-Person Team Architecture & Division of Responsibilities

To ensure academic rigor, clean separation of concerns, and reproducible evaluation, work is divided across 3 dedicated team member roles:

| Role | Member | Responsibilities | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Member 1** | **My Responsibility** | Dataset selection & citation, deterministic preprocessing, seed control (`42`), stratified Train/Val/Test splitting, custom vocabulary/tokenizer from scratch, **Model A (Custom Transformer from scratch)**, Colab T4 training pipeline, Model A test evaluation, and standardized inference interface for Member 3. | `data/train.csv`, `data/val.csv`, `data/test.csv`, `models/scratch/`, `notebooks/train_model_a_colab.ipynb`, `scripts/train_scratch.py`, `artifacts/model_a_metrics.json` |
| **Member 2** | *Collaborator* | **Model B (Pretrained Transformer)** selection, fine-tuning pipeline on the exact same `data/train.csv` and `data/val.csv`, evaluation on the exact same held-out `data/test.csv`, and Model B inference adapter. | `models/pretrained/`, `scripts/train_pretrained.py`, `artifacts/model_b_metrics.json` |
| **Member 3** | *Collaborator* | **Streamlit Comparison Dashboard** (`app.py`), side-by-side interactive inference, visual latency and metric benchmarking, error analysis visualization, and project report synthesis. | `app.py`, interactive comparison charts, final report |

---

## 🏛️ Project Directory Structure

```text
transformer-spam-toxicity-comparison/
│
├── data/                                 # Datasets & split artifacts
│   ├── raw/                              # Untouched raw dataset downloads (.gitkeep)
│   ├── processed/                        # Intermediate processed files (.gitkeep)
│   ├── train.csv                         # Deterministic training split (fit tokenizer strictly here)
│   ├── val.csv                           # Validation split (hyperparameter tuning & checkpointing)
│   ├── test.csv                          # Held-out test split (untouched until final evaluation)
│   └── dataset_metadata.json             # Dataset provenance, license, citation, and distribution
│
├── models/
│   ├── scratch/                          # [MEMBER 1] Custom Transformer from Scratch
│   │   ├── __init__.py
│   │   ├── transformer.py                # Pure PyTorch primitives: Self-Attention, FFN, Positional, etc.
│   │   ├── tokenizer.py                  # Tokenizer trained strictly on train.csv
│   │   └── inference.py                  # Standardized ScratchModelPredictor contract
│   │
│   └── pretrained/                       # [MEMBER 2 PLACEHOLDER] Pretrained Transformer
│       ├── __init__.py
│       ├── model.py                      # Hugging Face pretrained classifier wrapper
│       ├── inference.py                  # Standardized PretrainedModelPredictor contract
│       └── README.md                     # Integration guidelines for Member 2
│
├── notebooks/
│   ├── README.md
│   └── train_model_a_colab.ipynb         # [MEMBER 1] Free Colab T4 training & evaluation notebook
│
├── scripts/
│   ├── __init__.py
│   ├── data_prep.py                      # Deterministic dataset downloading & stratified splitting
│   ├── train_scratch.py                  # Standalone CLI training script for Model A
│   └── evaluate.py                       # Evaluation script generating standardized metrics JSON
│
├── artifacts/                            # Weights, logs, metrics, and visualization artifacts
│   ├── model_a_scratch.pt                # Trained Model A weights (Git-ignored)
│   ├── scratch_tokenizer.json            # Vocabulary and tokenizer serialization
│   ├── model_a_metrics.json              # Accuracy, Macro-F1, Precision, Recall, Latency, Params
│   ├── model_a_confusion_matrix.png      # Confusion matrix plot
│   └── README.md
│
├── app.py                                # [MEMBER 3 PLACEHOLDER] Streamlit comparative dashboard
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Clean repo hygiene rules
└── README.md                             # Documentation
```

---

## 📐 Strict Model A Architectural Rules (Member 1 Grading Mandate)

Model A is implemented strictly from scratch using fundamental PyTorch building blocks (`nn.Linear`, `nn.Embedding`, `nn.Dropout`, `nn.LayerNorm`, tensor arithmetic):
- 🚫 **Forbidden:** `torch.nn.Transformer`, `torch.nn.TransformerEncoder`, `torch.nn.TransformerEncoderLayer`, `torch.nn.MultiheadAttention`.
- 🚫 **Forbidden:** Pretrained embeddings (e.g. GloVe, Word2Vec) or pretrained checkpoints (all weights randomly initialized).
- ✅ **Required Components:**
  1. Token Embeddings + Sinusoidal / Learned Positional Encodings
  2. Multi-Head Self-Attention with explicit $Q, K, V$ linear projections, scaled dot-product attention ($\frac{QK^T}{\sqrt{d_k}}$), attention masking for `<PAD>`, and head concatenation
  3. Position-wise Feed-Forward Networks ($Linear \to GELU/ReLU \to Linear$)
  4. Pre-LN or Post-LN Residual Connections with Layer Normalization
  5. Padding mask handling (ensuring pad tokens never receive attention or skew pooling)
  6. Non-pad sequence pooling (mean pooling or `[CLS]` token) + Linear Classification Head

---

## 🔌 Standardized Inference Contract (Member 1 ➡️ Member 3 Handoff)

Member 1 exposes `ScratchModelPredictor` in `models/scratch/inference.py`:

```python
class ScratchModelPredictor:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cpu"):
        ...

    def predict(self, text: str) -> dict:
        """
        Returns:
            {
                "model_name": "Model A (Custom Transformer - From Scratch)",
                "label": "Spam",              # str
                "confidence": 0.9842,         # float
                "probabilities": {            # dict[str, float]
                    "Ham": 0.0158,
                    "Spam": 0.9842
                },
                "latency_ms": 1.45            # float
            }
        """
        ...
```

---

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/amrrtarrek/transformer-spam-toxicity-comparison.git
cd transformer-spam-toxicity-comparison

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
