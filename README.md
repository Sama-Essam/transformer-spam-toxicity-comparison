# Transformer: From Scratch vs. Pretrained Model
### Undergraduate NLP Course Project — Spam & Toxicity Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io/)

A modular comparative NLP study investigating the performance, computational efficiency, parameter footprint, and inference latency between:
1. **Model A (Member 1):** A custom Encoder-only Transformer classifier implemented **completely from scratch** using raw PyTorch primitives (no `nn.Transformer` or pretrained weights).
2. **Model B (Member 2):** A fine-tuned modern Pretrained Transformer (e.g., DistilBERT / BERT / RoBERTa via Hugging Face).
3. **Interactive Comparison UI (Member 3):** A unified Streamlit dashboard benchmarking both models side-by-side.

---

## 👥 3-Person Team Architecture & Division of Responsibilities

To ensure academic rigor, clean separation of concerns, and reproducible evaluation, work is divided across 3 dedicated team member roles:

| Role | Member | Responsibilities | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Current Scope** | Dataset selection & citation, deterministic preprocessing, seed control (`42`), stratified Train/Val/Test splitting, custom vocabulary/tokenizer from scratch, **Model A (Custom Transformer from scratch)**, Colab T4 training pipeline, Model A test evaluation, and standardized inference interface for Member 3. | `data/train.csv`, `data/val.csv`, `data/test.csv`, `models/scratch/`, `notebooks/train_model_a_colab.ipynb`, `scripts/train_scratch.py`, `artifacts/model_a_metrics.json` |
| **Member 2** | *Collaborator* | **Model B (Pretrained Transformer)** selection, fine-tuning pipeline on the exact same `data/train.csv` and `data/val.csv`, evaluation on the exact same held-out `data/test.csv`, and Model B inference adapter. | `models/pretrained/`, `scripts/train_pretrained.py`, `artifacts/model_b_metrics.json` |
| **Member 3** | *Collaborator* | **Streamlit Comparison Dashboard** (`app.py`), side-by-side interactive inference, visual latency and metric benchmarking, error analysis visualization, and project report synthesis. | `app.py`, interactive comparison charts, final report |

---

## 📊 Dataset Provenance, Metadata & Splits

- **Dataset Name:** UCI SMS Spam Collection
- **Task:** Binary Text Classification (Ham = 0, Spam = 1)
- **Source:** [UCI Machine Learning Repository (ID: 228)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Citation:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* ACM DocEng 2011.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Random Seed:** Fixed deterministic `SEED = 42` across all splits.
- **Subset Justification:** With 5,572 real-world SMS messages, moderate sequence lengths, and realistic class imbalance (~86.6% Ham, ~13.4% Spam), the dataset allows an Encoder-only Transformer trained from scratch to converge reliably within minutes on a free Google Colab T4 GPU, providing clear separation of self-attention quality without GPU memory exhaustion or excessive compute overhead.

### Deterministic Stratified Split Distribution

| Split | File | Total Samples | Ham (0) | Spam (1) | % Ham | % Spam |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (80%)** | `data/train.csv` | 4,457 | 3,860 | 597 | 86.61% | 13.39% |
| **Val (10%)** | `data/val.csv` | 556 | 482 | 74 | 86.69% | 13.31% |
| **Test (10%)** | `data/test.csv` | 559 | 483 | 76 | 86.40% | 13.60% |
| **Overall** | `data/raw/` | 5,572 | 4,825 | 747 | 86.59% | 13.41% |

*Note: Member 2 must evaluate Model B on the **exact same held-out `data/test.csv`** to guarantee fair scientific benchmarking.*

---

## 📐 Strict Model A Architectural Rules (Member 1 Grading Mandate)

Model A is implemented strictly from scratch using fundamental PyTorch building blocks (`nn.Linear`, `nn.Embedding`, `nn.Dropout`, `nn.LayerNorm`, tensor arithmetic):
- 🚫 **Forbidden:** `torch.nn.Transformer`, `torch.nn.TransformerEncoder`, `torch.nn.TransformerEncoderLayer`, `torch.nn.MultiheadAttention`.
- 🚫 **Forbidden:** Pretrained embeddings (e.g. GloVe, Word2Vec) or pretrained checkpoints (all weights randomly initialized).
- ✅ **Required Components:**
  1. Token Embeddings + Sinusoidal Positional Encodings
  2. Multi-Head Self-Attention with explicit $Q, K, V$ linear projections, scaled dot-product attention ($\frac{QK^T}{\sqrt{d_k}}$), attention masking for `<PAD>`, and head concatenation
  3. Position-wise Feed-Forward Networks ($Linear \to GELU/ReLU \to Linear$)
  4. Pre-LN Residual Connections with Layer Normalization
  5. Padding mask handling (ensuring pad tokens never receive attention or skew pooling)
  6. Masked mean pooling over non-pad tokens + Linear Classification Head

---

## 🏆 Model A Benchmark Results (Held-Out Test Set)

Evaluated strictly on `data/test.csv` (559 samples):

| Metric | Model A (From Scratch) |
| :--- | :--- |
| **Test Accuracy** | **98.21%** |
| **Test Macro-F1** | **0.9615** |
| **Macro Precision** | **96.68%** |
| **Macro Recall** | **95.64%** |
| **Ham F1-Score** | **0.9897** (Precision: 98.76%, Recall: 99.17%) |
| **Spam F1-Score** | **0.9333** (Precision: 94.59%, Recall: 92.11%) |
| **Avg. Single-Sample Latency** | **5.24 ms** |
| **Trainable Parameters** | **766,210** |
| **Confusion Matrix** | `[[479, 4], [6, 70]]` |

Confusion matrix plot saved to `artifacts/model_a_confusion_matrix.png`.

---

## 🔌 Standardized Inference Contract (Member 1 ➡️ Member 3 Handoff)

Member 1 exposes `ScratchModelPredictor` in `models/scratch/inference.py`:

```python
from models.scratch.inference import ScratchModelPredictor

predictor = ScratchModelPredictor(
    model_path="artifacts/model_a_scratch.pt",
    tokenizer_path="artifacts/scratch_tokenizer.json"
)

result = predictor.predict("Free entry in 2 a wkly comp to win a £1000 cash prize!")
print(result)
```

**Standardized Output Schema:**
```json
{
  "model_name": "Model A (Custom Transformer - From Scratch)",
  "label": "Spam",
  "confidence": 0.9999,
  "probabilities": {
    "Ham": 0.0001,
    "Spam": 0.9999
  },
  "latency_ms": 5.81
}
```

---

## 🚀 Execution & Quickstart

### 1. Local Setup
```bash
# Clone the repository
git clone https://github.com/amrrtarrek/transformer-spam-toxicity-comparison.git
cd transformer-spam-toxicity-comparison

# Install dependencies
pip install -r requirements.txt

# Run data preparation & deterministic splitting
python scripts/data_prep.py

# Train Model A
python scripts/train_scratch.py --epochs 15 --batch_size 64 --lr 3e-4

# Evaluate on held-out test set
python scripts/evaluate.py
```

### 2. Google Colab (Free T4 GPU)
Open [`notebooks/train_model_a_colab.ipynb`](notebooks/train_model_a_colab.ipynb) directly in Google Colab:
1. Upload the notebook to Google Colab.
2. Under **Runtime** -> **Change runtime type**, select **T4 GPU**.
3. Click **Runtime** -> **Run All** to execute data preparation, training, evaluation, and interactive inference.
