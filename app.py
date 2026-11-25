"""
Streamlit Comparison Dashboard (Spam & Toxicity Detection).
Interactive web application comparing:
- Model A: Custom PyTorch Transformer built from scratch (Member 1)
- Model B: Pretrained Hugging Face Transformer (DistilBERT - Member 2)
"""

import os
import json
import pandas as pd
import streamlit as st

from models.scratch.inference import ScratchModelPredictor
from models.pretrained.inference import PretrainedModelPredictor

st.set_page_config(
    page_title="Spam / Toxicity Detection: Scratch vs Pretrained",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Transformer: From Scratch vs. Pretrained Model")
st.markdown(
    """
    ### Undergraduate NLP Course Project
    **Task:** Spam / Toxicity Detection (Text Classification)
    
    ---
    #### Team Roles & Deliverables:
    - **Member 1:** Custom Transformer from scratch (PyTorch primitives only), deterministic dataset preparation, tokenizer, Colab training pipeline, and evaluation metrics.
    - **Member 2:** Pretrained Hugging Face model fine-tuning (DistilBERT), unified evaluation pipeline, and comparative benchmarking.
    - **Member 3:** Interactive Streamlit dashboard comparing inference latency, accuracy, parameter count, and prediction confidence.
    """
)

# Cached predictors for responsive real-time inference
@st.cache_resource
def load_predictors():
    pred_a, pred_b = None, None
    try:
        if os.path.exists("artifacts/model_a_scratch.pt"):
            pred_a = ScratchModelPredictor(
                model_path="artifacts/model_a_scratch.pt",
                tokenizer_path="artifacts/scratch_tokenizer.json"
            )
    except Exception as e:
        st.sidebar.error(f"Error loading Model A: {e}")

    try:
        if os.path.exists("artifacts/model_b_pretrained"):
            pred_b = PretrainedModelPredictor(
                model_path_or_name="artifacts/model_b_pretrained"
            )
    except Exception as e:
        st.sidebar.error(f"Error loading Model B: {e}")

    return pred_a, pred_b

predictor_a, predictor_b = load_predictors()

# User Input Box
st.subheader("📝 Interactive Live Inference")
user_text = st.text_area(
    "Input Text for Classification (SMS / Message)",
    "Free entry in 2 a wkly comp to win a £1000 cash prize! Text CLAIM to 87121.",
    height=90
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model A: Custom Transformer (From Scratch)")
    st.caption("Developed by Member 1 (766K parameters, no pretrained weights)")
    if st.button("Run Model A Inference", key="btn_model_a"):
        if predictor_a is not None:
            res_a = predictor_a.predict(user_text)
            st.success(f"**Predicted Label:** {res_a['label']} (Confidence: {res_a['confidence']*100:.2f}%)")
            st.write(f"⏱️ **Latency:** {res_a['latency_ms']:.2f} ms")
            st.json(res_a["probabilities"])
        else:
            st.warning("Model A checkpoint not found at 'artifacts/model_a_scratch.pt'. Run scripts/train_scratch.py.")

with col2:
    st.subheader("Model B: Pretrained Transformer (DistilBERT)")
    st.caption("Fine-tuned by Member 2 (67M parameters, Hugging Face DistilBERT)")
    if st.button("Run Model B Inference", key="btn_model_b"):
        if predictor_b is not None:
            res_b = predictor_b.predict(user_text)
            st.success(f"**Predicted Label:** {res_b['label']} (Confidence: {res_b['confidence']*100:.2f}%)")
            st.write(f"⏱️ **Latency:** {res_b['latency_ms']:.2f} ms")
            st.json(res_b["probabilities"])
        else:
            st.warning("Model B checkpoint not found at 'artifacts/model_b_pretrained'. Run scripts/train_pretrained.py.")

# Benchmark Table & Visualizations Section
st.markdown("---")
st.subheader("📊 Side-by-Side Test Benchmark Summary (Held-Out Test Set: 559 samples)")

comp_table_path = "results/comparison_table.csv"
if os.path.exists(comp_table_path):
    df_comp = pd.read_csv(comp_table_path)
    st.dataframe(df_comp, use_container_width=True)
else:
    st.info("Run `python scripts/evaluate.py` to generate the comparison table.")

col_img1, col_img2 = st.columns(2)
with col_img1:
    if os.path.exists("results/metrics_comparison.png"):
        st.image("results/metrics_comparison.png", caption="Overall Metrics & Parameter Benchmark")

with col_img2:
    if os.path.exists("results/confusion_matrices_comparison.png"):
        st.image("results/confusion_matrices_comparison.png", caption="Side-by-Side Test Confusion Matrices")
