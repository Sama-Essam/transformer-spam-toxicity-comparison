"""
Streamlit Comparison Dashboard (Member 3 Placeholder)
Interactive web application comparing:
- Model A: Custom PyTorch Transformer built from scratch (Member 1)
- Model B: Pretrained Hugging Face Transformer (Member 2)
"""

import streamlit as st

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
    - **Member 1 (Current Role):** Custom Transformer from scratch (PyTorch primitives only), deterministic dataset preparation, tokenizer, Colab training pipeline, and evaluation metrics.
    - **Member 2:** Pretrained Hugging Face model fine-tuning & evaluation.
    - **Member 3:** Interactive Streamlit dashboard comparing inference latency, accuracy, parameter count, and prediction confidence.
    """
)

st.info("Member 3 dashboard placeholder. The interactive comparison interface will be developed here.")

user_text = st.text_area(
    "Input Text for Classification",
    "Free entry in 2 a wkly comp to win a £1000 cash prize! Text CLAIM to 87121.",
    height=100
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model A: Custom Transformer (From Scratch)")
    st.caption("Developed by Member 1")
    if st.button("Run Model A Inference", key="btn_model_a"):
        st.warning("Model A weights pending training. Run Phase 1 training to generate artifacts.")

with col2:
    st.subheader("Model B: Pretrained Transformer (Hugging Face)")
    st.caption("Developed by Member 2")
    if st.button("Run Model B Inference", key="btn_model_b"):
        st.warning("Model B placeholder. To be implemented by Member 2.")
