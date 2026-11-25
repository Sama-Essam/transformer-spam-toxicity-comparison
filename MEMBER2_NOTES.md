# 📘 Member 2 Technical Documentation & Notes: Pretrained Transformer & Unified Evaluation

**Course Project:** Transformer — From Scratch vs. Pretrained Model  
**Task:** Binary Spam and Toxicity Detection  
**Author / Role:** Member 2 (Pretrained Transformer Modeling & Unified Comparative Benchmarking)  
**Evaluated Dataset:** UCI SMS Spam Collection (`data/test.csv`, 559 held-out test samples)  
**Deterministic Seed:** `SEED = 42` across all data loaders, model initializations, and training splits.

---

## 1. Pretrained Checkpoint Selection & Technical Justification

### Selected Model: `distilbert-base-uncased`
- **Source / Citation:** Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108.
- **Backbone Architecture:** 6-layer, 768-hidden dimension Transformer encoder with 12 multi-head self-attention heads (distilled from `bert-base-uncased` via knowledge distillation).
- **Tokenization:** WordPiece tokenizer with 30,522 vocabulary items, lowercase normalization, and subword segmentation (`AutoTokenizer.from_pretrained('distilbert-base-uncased')`).
- **Why DistilBERT?**
  1. **Parameter Efficiency:** Contains **~66.96 million parameters**, which is 40% smaller than BERT-Base (110M), making fine-tuning and inference dramatically faster while retaining 97% of BERT's language understanding capabilities.
  2. **Subword Robustness for Spam & Toxicity:** Spam and toxic messages frequently exhibit deliberate character substitutions, leetspeak, concatenated words, and punctuation evasion (e.g., `w!n`, `£1000`, `CLAIM`). WordPiece subword tokenization naturally fragments unseen obfuscations into known subword roots, completely avoiding the catastrophic Out-of-Vocabulary (OOV) bottlenecks inherent in word-level tokenizers.
  3. **Richer Bidirectional Context:** Pretrained on Toronto BookCorpus and English Wikipedia (3.3 billion tokens), providing rich contextual representations of English semantics and syntax before seeing a single domain sample.

---

## 2. Experimental Setup, Hyperparameters & Training Alignment

To ensure a scientifically fair and valid comparison against Member 1's Model A:
- **Strict Data Integrity:** Model B was fine-tuned strictly on Member 1's `data/train.csv` (4,457 samples, 80%) and validated on `data/val.csv` (556 samples, 10%). The held-out `data/test.csv` (559 samples, 10%) was **never touched** or tuned on during training.
- **Hardware Environment:** Multi-core CPU platform (also compatible with GPU/T4 execution via `--device`).
- **Fine-Tuning Hyperparameters:**
  - **Optimizer:** `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
  - **Learning Rate:** $3 \times 10^{-5}$
  - **Weight Decay:** $0.01$ (with bias and LayerNorm parameters excluded from decay)
  - **Learning Rate Schedule:** Linear warmup over 10% of total training steps, followed by linear decay to 0
  - **Batch Size:** $32$
  - **Maximum Sequence Length:** $128$ tokens (matching Model A)
  - **Loss Function:** Class-weighted Cross-Entropy Loss ($\text{weight}_{\text{Ham}} = 0.58$, $\text{weight}_{\text{Spam}} = 3.73$) to counteract the ~86.6% / 13.4% class imbalance.
  - **Epochs:** $2$ epochs (sufficient for convergence without catastrophic forgetting on downstream classification)
  - **Measured Fine-Tuning Duration:** **2,111.49 seconds (~35.19 minutes on CPU)**. On an NVIDIA T4 GPU, this translates to ~25-35 seconds.

---

## 3. Side-by-Side Test Benchmark Results

Both Model A and Model B were benchmarked on the exact same 559 held-out test samples (`data/test.csv`). Single-sample latency was measured across all individual inference calls to emulate production deployment.

| Metric | Model A: From Scratch (Member 1) | Model B: Pretrained DistilBERT (Member 2) | Benchmark Delta / Advantage |
| :--- | :---: | :---: | :---: |
| **Model Architecture** | Custom Transformer (2 Layers, 4 Heads) | DistilBERT (`distilbert-base-uncased`) | Distilled Pretrained Backbone |
| **Pretrained Weights?** | ❌ No (Randomly Initialized) | ✅ Yes (Wiki + BookCorpus Pretrained) | Transfer Learning |
| **Tokenizer Type** | Custom Word-level Regex Tokenizer | WordPiece Subword Tokenizer | Robust Subword Tokenizer |
| **Vocabulary Size** | 3,789 tokens (fitted strictly on train) | 30,522 subword tokens | 8x richer subword coverage |
| **Trainable Parameters** | **766,210 (~0.77M)** | **66,955,010 (~66.96M)** | Model A is **87.4x smaller** |
| **Held-Out Test Accuracy** | **98.21%** | **99.11%** | **+0.90% (Model B leads)** |
| **Macro-F1 Score** | **0.9611** | **0.9809** | **+0.0198 (Model B leads)** |
| **Macro Precision** | **97.20%** | **98.36%** | **+1.16%** |
| **Macro Recall** | **95.08%** | **97.82%** | **+2.74% (Substantial gain)** |
| **Ham F1-Score (Majority)** | **0.9897** | **0.9948** | +0.0051 |
| **Spam F1-Score (Minority)** | **0.9324** | **0.9669** | **+0.0345 (Significant improvement)** |
| **Ham Precision / Recall** | 98.56% / 99.38% | 99.38% / 99.59% | Fewer False Positives |
| **Spam Precision / Recall** | 95.83% / 90.79% | 97.33% / 96.05% | **Recall jump: +5.26% on Spam** |
| **Test Confusion Matrix** | `[[480, 3], [7, 69]]` | `[[481, 2], [3, 73]]` | Model B cuts Spam misses from 7 down to 3 |
| **False Positives (Ham classified as Spam)** | 3 / 483 (0.62%) | **2 / 483 (0.41%)** | Lower user-facing annoyance |
| **False Negatives (Spam missed)** | 7 / 76 (9.21%) | **3 / 76 (3.95%)** | **57% reduction in missed spam** |
| **Avg. Single-Sample Latency (CPU)** | **2.14 ms** | **67.59 ms** | Model A is **31.6x faster** |
| **Training / Fine-Tuning Duration** | 7.93 s (GPU) / 814 s (CPU) | 2,111 s (CPU) / ~30 s (GPU) | Model A converges in fewer FLOPs |

---

## 4. In-Depth Technical Analysis & Insights

### 4.1 The Advantage of Pretraining: Semantic Sensitivity & Low-Resource Recall
1. **The Minority Class Challenge:** In realistic spam/toxicity detection, class imbalance is a fundamental barrier (~86.6% Ham vs ~13.4% Spam). A model trained from scratch must learn syntactic parsing, token semantic affinities, and task-specific spam triggers simultaneously from only 597 positive training instances.
2. **Prior World Knowledge:** Model B leverages pre-acquired linguistic representations from billions of pretraining tokens. It already understands linguistic subtleties (e.g., urgency markers, deceptive phrasing, financial keywords, informal syntax) and requires only minimal parameter adjustments (gradient alignment) to map these representations to the binary decision boundary.
3. **Drastic Reduction in False Negatives:** The strongest empirical benefit of pretraining is in **Spam Recall (96.05% vs 90.79%)**. Model A missed 7 spam messages out of 76 (a 9.21% miss rate). Model B missed only 3 spam messages (a 3.95% miss rate) — cutting the false negative rate by **more than half (57% reduction)**.

### 4.2 Error Analysis: Why Model A Missed Certain Samples
- **Subword Tokenization Evasion:** Model A relies on a word-level regex tokenizer (`SimpleTokenizer`). When spam messages introduce novel obfuscations, misspelled numbers, or unconventional punctuation (e.g., `C-A-L-L`, `08712478576`, `£1000cash`), Model A frequently maps them to `<UNK>`, losing critical discriminative signal.
- **Context Length & Sparse Attention:** In a 2-layer Transformer initialized from scratch, self-attention maps over short datasets tend to concentrate heavily on high-frequency unigrams rather than long-range dependency interactions. Model B's deep 6-layer 12-head attention captures contextual nuances even when keywords are disguised.

### 4.3 Computational Trade-offs: When to Deploy Model A vs. Model B
- **Parameter Footprint:** Model A requires only **766,210 parameters (3.1 MB on disk)** compared to Model B's **66,955,010 parameters (268 MB on disk)** — an 87.4x difference in memory overhead.
- **Inference Latency:** On standard CPU hardware, Model A executes inference in **2.14 ms**, while Model B requires **67.59 ms** (a 31.6x latency factor).
- **Engineering Recommendation:**
  - If deploying on ultra-constrained edge devices (microcontrollers, mobile phones, low-power IoT gateways) where sub-5ms latency and sub-10MB RAM are strict constraints, **Model A** is a viable, high-performing solution (achieving >98% accuracy with minimal overhead).
  - If deploying in cloud security gateways, production email/SMS servers, or moderation APIs where missing a toxic attack or phishing link has high financial/security repercussions, **Model B (Pretrained DistilBERT)** is unequivocally superior due to its higher Macro-F1 (0.9809) and 57% reduction in false negative risk.

---

## 5. Artifacts & Deliverables Summary for Team Handoff

| File Path | Description | Intended Consumer |
| :--- | :--- | :--- |
| `models/pretrained/model.py` | `PretrainedTransformerClassifier` class wrapping Hugging Face DistilBERT | Member 2 & 3 |
| `models/pretrained/inference.py` | `PretrainedModelPredictor` and `predict_pretrained(text)` function | Member 3 (`app.py`) |
| `pretrained_model.py` | Root-level convenience wrapper for Model B inference | Member 3 / CLI |
| `scripts/train_pretrained.py` | Deterministic Model B fine-tuning script (`SEED=42`, zero test leakage) | Reproducibility |
| `scripts/evaluate.py` | Unified test evaluation script evaluating both models side-by-side | Team / Grader |
| `artifacts/model_b_pretrained/` | Saved weights (`model.safetensors`), configuration (`config.json`), and tokenizer | Member 3 (`app.py`) |
| `artifacts/model_b_metrics.json` | Detailed Model B test set metrics | Dashboard & Report |
| `artifacts/model_b_confusion_matrix.png` | Model B test set confusion matrix visualization | Report & Slides |
| `results/comparison_metrics.json` | Complete side-by-side hierarchical metrics JSON | Member 3 Dashboard |
| `results/comparison_table.csv` | Clean side-by-side comparison CSV table | Final Report |
| `results/metrics_comparison.png` | 4-panel publication-grade comparative chart | Report & Dashboard |
| `results/confusion_matrices_comparison.png`| Dual-panel comparative confusion matrix | Report & Dashboard |
| `requirements.txt` | Fully pinned dependencies including `transformers`, `datasets`, `accelerate` | Environment Setup |
