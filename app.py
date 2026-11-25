"""
Streamlit Comparison Dashboard (Member 3 deliverable).

Spam detection: Model A (custom Transformer from scratch, Member 1)
vs Model B (fine-tuned DistilBERT, Member 2) on the UCI SMS Spam Collection.

Run locally:  streamlit run app.py
The app degrades gracefully: if a model checkpoint is missing, that model's
panel shows instructions, while benchmarks/dataset tabs still work.
"""

import json
import os
import time

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # relative artifact paths used by the predictors

st.set_page_config(page_title="Scratch vs Pretrained Transformer", page_icon="🛡️", layout="wide")

MODEL_A_PATH = "artifacts/model_a_scratch.pt"
TOKENIZER_A_PATH = "artifacts/scratch_tokenizer.json"
MODEL_B_PATH = "artifacts/model_b_pretrained"

EXAMPLES = {
    "Prize scam": "Free entry in 2 a wkly comp to win a £1000 cash prize! Text CLAIM to 87121 now.",
    "Normal message": "Hey, are we still meeting for lunch at 1? I'll be a few minutes late.",
    "Obfuscated spam": "W!N a FR33 iPhone!! C-A-L-L 08712478576 now, claim your £1000cash",
    "Tricky ham": "Your parcel is at the front desk, call me when you get back.",
    "Toxic-style insult (out of domain)": "You are a complete idiot and nobody wants you here.",
}


# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading Model A...")
def load_model_a():
    if not (os.path.exists(MODEL_A_PATH) and os.path.exists(TOKENIZER_A_PATH)):
        return None, f"Checkpoint not found at `{MODEL_A_PATH}`. Run `python scripts/train_scratch.py` (or the Colab notebook) and copy the .pt file into `artifacts/`."
    try:
        from models.scratch.inference import ScratchModelPredictor
        return ScratchModelPredictor(model_path=MODEL_A_PATH, tokenizer_path=TOKENIZER_A_PATH), None
    except Exception as e:  # noqa: BLE001
        return None, f"Could not load Model A: {e}"


@st.cache_resource(show_spinner="Loading Model B (DistilBERT)...")
def load_model_b():
    if not os.path.exists(MODEL_B_PATH):
        return None, f"Fine-tuned checkpoint not found at `{MODEL_B_PATH}/`. Run `python scripts/train_pretrained.py` first."
    try:
        from models.pretrained.inference import PretrainedModelPredictor
        return PretrainedModelPredictor(model_path_or_name=MODEL_B_PATH), None
    except Exception as e:  # noqa: BLE001
        return None, f"Could not load Model B: {e}"


@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def exists(p):
    return os.path.exists(p)


pred_a, err_a = load_model_a()
pred_b, err_b = load_model_b()


def run_model(pred, text, warm=True):
    """Predict; one warm-up call so latency isn't dominated by first-call overhead."""
    if warm:
        pred.predict("warm up")
    return pred.predict(text)


def result_card(res, container):
    label = res["label"]
    (container.error if label == "Spam" else container.success)(
        f"**{label}** · confidence {res['confidence'] * 100:.2f}%"
    )
    container.metric("Latency", f"{res['latency_ms']:.2f} ms")
    container.bar_chart(pd.Series(res["probabilities"], name="probability"), height=160)


# ---------------------------------------------------------------- header
st.title("🛡️ Transformer: From Scratch vs. Pretrained")
st.caption("NLP course project · spam detection · UCI SMS Spam Collection · seed 42 · identical held-out test set for both models")

with st.sidebar:
    st.header("Model status")
    st.write("Model A (scratch):", "✅ loaded" if pred_a else "⚠️ unavailable")
    if err_a:
        st.caption(err_a)
    st.write("Model B (DistilBERT):", "✅ loaded" if pred_b else "⚠️ unavailable")
    if err_b:
        st.caption(err_b)
    st.divider()
    st.caption("Model A: Member 1 · Model B: Member 2 · Dataset, app & submission: Member 3")

tab_live, tab_bench, tab_test, tab_data, tab_about = st.tabs(
    ["🔎 Live comparison", "📊 Benchmarks", "🧪 Test-set explorer", "📁 Dataset", "ℹ️ About"]
)

# ---------------------------------------------------------------- live
with tab_live:
    st.subheader("Side-by-side inference")
    choice = st.selectbox("Load an example", ["(custom)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "Congratulations! You won a $1,000 gift card. Click here to claim.")
    text = st.text_area("Message", default, height=100, key=f"text_{choice}")

    if st.button("Run both models", type="primary", disabled=not text.strip()):
        c1, c2 = st.columns(2)
        results = {}
        for col, name, pred, err in (
            (c1, "Model A · Custom Transformer (scratch)", pred_a, err_a),
            (c2, "Model B · DistilBERT (pretrained)", pred_b, err_b),
        ):
            col.markdown(f"#### {name}")
            if pred is None:
                col.warning(err)
                continue
            res = run_model(pred, text)
            results[name] = res
            result_card(res, col)
        if len(results) == 2:
            a, b = list(results.values())
            if a["label"] == b["label"]:
                st.info(f"Both models agree: **{a['label']}**.")
            else:
                st.warning(f"Models **disagree**: A says **{a['label']}**, B says **{b['label']}**.")
            st.caption(f"Latency ratio B/A on this input: {b['latency_ms'] / max(a['latency_ms'], 1e-9):.1f}×")
    st.caption("Both models were trained on SMS spam/ham only, so toxic-text examples are out of domain and results there are not meaningful.")

# ---------------------------------------------------------------- benchmarks
with tab_bench:
    st.subheader("Held-out test benchmark (559 messages)")
    cmp_path = "results/comparison_metrics.json"
    if exists(cmp_path):
        cmp = load_json(cmp_path)
        A, B = cmp["model_a"], cmp["model_b"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy A → B", f"{A['accuracy']*100:.2f}% → {B['accuracy']*100:.2f}%", f"{(B['accuracy']-A['accuracy'])*100:+.2f} pts")
        k2.metric("Macro-F1 A → B", f"{A['macro_f1']:.4f} → {B['macro_f1']:.4f}", f"{B['macro_f1']-A['macro_f1']:+.4f}")
        k3.metric("Latency A → B", f"{A['average_inference_latency_ms']:.2f} → {B['average_inference_latency_ms']:.2f} ms")
        k4.metric("Params A → B", f"{A['trainable_parameters']/1e6:.2f}M → {B['trainable_parameters']/1e6:.1f}M")

        rows = {
            "Accuracy": (A["accuracy"], B["accuracy"]),
            "Macro-F1": (A["macro_f1"], B["macro_f1"]),
            "Macro precision": (A["macro_precision"], B["macro_precision"]),
            "Macro recall": (A["macro_recall"], B["macro_recall"]),
            "Spam F1": (A["class_metrics"]["Spam"]["f1_score"], B["class_metrics"]["Spam"]["f1_score"]),
            "Spam recall": (A["class_metrics"]["Spam"]["recall"], B["class_metrics"]["Spam"]["recall"]),
        }
        df = pd.DataFrame(rows, index=["Model A (scratch)", "Model B (DistilBERT)"]).T
        left, right = st.columns(2)
        left.markdown("**Quality metrics**")
        left.bar_chart(df, height=320)
        right.markdown("**Cost: latency (ms) and parameters (M)**")
        cost = pd.DataFrame(
            {
                "Latency (ms)": [A["average_inference_latency_ms"], B["average_inference_latency_ms"]],
                "Parameters (M)": [A["trainable_parameters"] / 1e6, B["trainable_parameters"] / 1e6],
            },
            index=["Model A (scratch)", "Model B (DistilBERT)"],
        )
        right.dataframe(cost.style.format("{:.2f}"), width="stretch")
        right.caption(
            f"B is {cmp['delta']['parameter_ratio']:.0f}× larger and {cmp['delta']['latency_ratio']:.0f}× slower per message "
            f"(CPU, single-sample)."
        )

        st.markdown("**Confusion matrices** (rows = true, columns = predicted)")
        cA, cB = st.columns(2)
        for col, name, M in ((cA, "Model A", A), (cB, "Model B", B)):
            cm = pd.DataFrame(M["confusion_matrix"], index=["True Ham", "True Spam"], columns=["Pred Ham", "Pred Spam"])
            col.markdown(f"*{name}*")
            col.dataframe(cm, width="stretch")
        st.caption(
            f"Training time: A {A['training_time_seconds']:.1f} s · B {B['training_time_seconds']:.0f} s "
            f"(hardware recorded as {A['hardware']} / {B['hardware']})."
        )
    else:
        st.info("Run `python scripts/evaluate.py` to generate `results/comparison_metrics.json`.")

    if exists("results/comparison_table.csv"):
        with st.expander("Full comparison table"):
            st.dataframe(load_csv("results/comparison_table.csv"), width="stretch")
    i1, i2 = st.columns(2)
    if exists("results/metrics_comparison.png"):
        i1.image("results/metrics_comparison.png", caption="Metrics and size benchmark")
    if exists("results/confusion_matrices_comparison.png"):
        i2.image("results/confusion_matrices_comparison.png", caption="Test confusion matrices")

# ---------------------------------------------------------------- test explorer
with tab_test:
    st.subheader("Browse the held-out test set")
    if not exists("data/test.csv"):
        st.info("`data/test.csv` not found. Run `python scripts/data_prep.py`.")
    else:
        test = load_csv("data/test.csv")
        f1, f2 = st.columns([1, 2])
        lab = f1.radio("Filter by true label", ["All", "Spam", "Ham"], horizontal=True)
        q = f2.text_input("Search text")
        view = test if lab == "All" else test[test["label"] == lab]
        if q:
            view = view[view["text"].str.contains(q, case=False, na=False)]
        st.caption(f"{len(view)} of {len(test)} messages")
        st.dataframe(view[["label", "text"]].reset_index(drop=True), width="stretch", height=260)

        st.markdown("**Error analysis** — run both models on the whole test set and list the mistakes")
        if pred_a is None or pred_b is None:
            st.info("Needs both model checkpoints loaded.")
        elif st.button("Run error analysis (559 messages)"):
            progress = st.progress(0.0)
            preds_a, preds_b = [], []
            for i, t in enumerate(test["text"]):
                preds_a.append(pred_a.predict(t)["label"])
                preds_b.append(pred_b.predict(t)["label"])
                if i % 20 == 0:
                    progress.progress((i + 1) / len(test))
            progress.empty()
            out = test.assign(pred_A=preds_a, pred_B=preds_b)
            acc_a, acc_b = (out.pred_A == out.label).mean(), (out.pred_B == out.label).mean()
            m1, m2 = st.columns(2)
            m1.metric("Model A accuracy (live)", f"{acc_a*100:.2f}%")
            m2.metric("Model B accuracy (live)", f"{acc_b*100:.2f}%")
            wrong = out[(out.pred_A != out.label) | (out.pred_B != out.label)]
            st.write(f"{len(wrong)} messages misclassified by at least one model:")
            st.dataframe(wrong[["label", "pred_A", "pred_B", "text"]].reset_index(drop=True), width="stretch")

# ---------------------------------------------------------------- dataset
with tab_data:
    st.subheader("Dataset provenance")
    if exists("data/dataset_metadata.json"):
        md = load_json("data/dataset_metadata.json")
        st.markdown(
            f"""
- **Name:** {md['dataset_name']}
- **Task:** {md['task']}
- **Source:** [{md['source_url']}]({md['source_url']})
- **Authors / citation:** {md['citation']}
- **License:** {md['license']}
- **Seed:** {md['random_seed']} · **Split:** 80 / 10 / 10 stratified
"""
        )
        sd = md["splits_distribution"]
        split_df = pd.DataFrame(
            {k.capitalize(): {"Ham": v["counts"]["Ham"], "Spam": v["counts"]["Spam"]} for k, v in sd.items()}
        )
        st.bar_chart(split_df.T, height=260)
        st.dataframe(split_df.T.assign(Total=split_df.sum().values), width="stretch")
        with st.expander("Why this dataset"):
            st.write(md["subset_justification"])
    else:
        st.info("`data/dataset_metadata.json` not found.")

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("Project summary")
    st.markdown(
        """
| | Model A | Model B |
|---|---|---|
| Owner | Member 1 | Member 2 |
| Architecture | 2-layer, 4-head encoder, written from raw PyTorch primitives | DistilBERT (`distilbert-base-uncased`) |
| Pretrained weights | None (random init) | Yes, fine-tuned |
| Tokenizer | Custom word-level regex | WordPiece |

**Requirements checklist:** self-implemented Model A · shared held-out test set · working Streamlit demo · dataset cited with license · seed and preprocessing recorded.
"""
    )
    st.caption("Run locally: `pip install -r requirements.txt && streamlit run app.py`")
