"""
app.py
------
Streamlit dashboard for the Financial News Sentiment project.

Run from the project root with:
    streamlit run app.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# Make project root importable (this file lives directly in the project root,
# alongside preprocessing.py, rnn_models.py, train_evaluate.py, models/, data/)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocessing import clean_text, Vocabulary  # noqa: E402
from rnn_models import SimpleRNNClassifier, LSTMClassifier, GRUClassifier  # noqa: E402

LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_COLOR = {"Bearish": "#E24B4A", "Bullish": "#3B9E5A", "Neutral": "#888780"}

MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

RNN_BUILDERS = {
    "LSTM": (LSTMClassifier, os.path.join(MODELS_DIR, "lstm_best.pt")),
    "GRU": (GRUClassifier, os.path.join(MODELS_DIR, "gru_best.pt")),
    "Simple RNN": (SimpleRNNClassifier, os.path.join(MODELS_DIR, "simplernn_best.pt")),
}
BERT_DIR = os.path.join(MODELS_DIR, "finbert_finetuned")

st.set_page_config(page_title="Financial Sentiment", page_icon="📈", layout="wide")


# ─────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────
@st.cache_resource
def load_vocab():
    vocab_path = os.path.join(MODELS_DIR, "vocab.pkl")
    if not os.path.exists(vocab_path):
        return None
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_rnn_model(model_name: str, _vocab_len: int):
    builder, path = RNN_BUILDERS[model_name]
    if not os.path.exists(path):
        return None
    model = builder(vocab_size=_vocab_len)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_bert_model():
    if not os.path.isdir(BERT_DIR):
        return None, None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
    model.eval()
    return model, tokenizer


@st.cache_data
def load_dataset():
    train_path = os.path.join(DATA_DIR, "sent_train.csv")
    valid_path = os.path.join(DATA_DIR, "sent_valid.csv")
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        return None, None
    return pd.read_csv(train_path), pd.read_csv(valid_path)


# ─────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────
def predict_rnn(text: str, model, vocab: Vocabulary) -> dict:
    cleaned = clean_text(text)
    ids = torch.tensor([vocab.encode(cleaned)], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return {"label": LABEL_MAP[pred_idx], "probs": probs}


def predict_bert(text: str, model, tokenizer) -> dict:
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return {"label": LABEL_MAP[pred_idx], "probs": probs}


def probs_bar_chart(probs) -> go.Figure:
    labels = [LABEL_MAP[i] for i in range(3)]
    colors = [LABEL_COLOR[l] for l in labels]
    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color=colors, text=[f"{p:.1%}" for p in probs],
                            textposition="outside"))
    fig.update_layout(yaxis_range=[0, 1], yaxis_title="Probability", height=320,
                       margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("📈 Financial Sentiment")
st.sidebar.caption("Bearish · Bullish · Neutral")

vocab = load_vocab()
bert_available = os.path.isdir(BERT_DIR)
available_rnn = [name for name, (_, path) in RNN_BUILDERS.items() if os.path.exists(path)] if vocab else []
available_models = available_rnn + (["FinBERT"] if bert_available else [])

if not available_models:
    st.sidebar.warning(
        "No trained models found yet.\n\n"
        "Train at least one:\n"
        "```\npython train_evaluate.py\n```\n"
        "or fine-tune FinBERT:\n"
        "```\npython bert_finetune.py\n```"
    )
    model_choice = None
else:
    model_choice = st.sidebar.selectbox("Model", available_models)

train_df, valid_df = load_dataset()
if train_df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset")
    st.sidebar.write(f"Train: **{len(train_df):,}** rows")
    st.sidebar.write(f"Valid: **{len(valid_df):,}** rows")
    dist = valid_df["label"].map(LABEL_MAP).value_counts()
    dist_fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values,
        marker_color=[LABEL_COLOR[l] for l in dist.index],
    ))
    dist_fig.update_layout(title="Validation class distribution", height=260,
                            margin=dict(l=10, r=10, t=40, b=10))
    st.sidebar.plotly_chart(dist_fig, use_container_width=True)
else:
    st.sidebar.info("Dataset not found in `data/`. Run `prepare_data.py` first.")

# ─────────────────────────────────────────────
# Main area — single prediction
# ─────────────────────────────────────────────
st.title("Financial News Sentiment")
st.write("Predict whether a financial headline / tweet is **Bearish**, **Bullish**, or **Neutral**.")

examples = [
    "$TSLA surges 15% after record quarterly earnings beat",
    "$BYND slides after guidance cut, analysts downgrade to sell",
    "$AAPL reports in line with expectations this quarter",
]
cols = st.columns(len(examples))
example_click = None
for c, ex in zip(cols, examples):
    if c.button(ex[:35] + "…", use_container_width=True):
        example_click = ex

text_input = st.text_area("Headline / tweet", value=example_click or "", height=100,
                           placeholder="e.g. $NVDA jumps after strong AI chip demand")

predict_clicked = st.button("Predict", type="primary", disabled=model_choice is None)

if predict_clicked and text_input.strip():
    if model_choice == "FinBERT":
        model, tokenizer = load_bert_model()
        result = predict_bert(text_input, model, tokenizer)
    else:
        model = load_rnn_model(model_choice, len(vocab))
        result = predict_rnn(text_input, model, vocab)

    label = result["label"]
    color = LABEL_COLOR[label]
    st.markdown(
        f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;background:{color}22;"
        f"border:1px solid {color};display:inline-block;font-weight:600;color:{color}'>"
        f"Prediction: {label}</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(probs_bar_chart(result["probs"]), use_container_width=True)
elif predict_clicked:
    st.warning("Enter some text first.")

# ─────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Batch prediction")
st.write("Upload a CSV with a `text` column to score many headlines at once.")

uploaded = st.file_uploader("CSV file", type=["csv"])
if uploaded is not None and model_choice is not None:
    batch_df = pd.read_csv(uploaded)
    if "text" not in batch_df.columns:
        st.error("CSV must contain a `text` column.")
    else:
        with st.spinner(f"Scoring {len(batch_df):,} rows with {model_choice}..."):
            if model_choice == "FinBERT":
                model, tokenizer = load_bert_model()
                preds = [predict_bert(t, model, tokenizer) for t in batch_df["text"]]
            else:
                model = load_rnn_model(model_choice, len(vocab))
                preds = [predict_rnn(t, model, vocab) for t in batch_df["text"]]

        batch_df["predicted_label"] = [p["label"] for p in preds]
        batch_df["bearish_prob"] = [round(float(p["probs"][0]), 4) for p in preds]
        batch_df["bullish_prob"] = [round(float(p["probs"][1]), 4) for p in preds]
        batch_df["neutral_prob"] = [round(float(p["probs"][2]), 4) for p in preds]

        st.dataframe(batch_df, use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            batch_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
elif uploaded is not None:
    st.warning("No trained model available to score this file yet.")