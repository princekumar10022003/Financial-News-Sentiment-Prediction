import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #00d2ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
        letter-spacing: 2px;
    }

    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2a4a, #2d3561);
        border: 1px solid #3d4f7c;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #2d3561);
        border-radius: 20px;
        padding: 30px;
        border: 2px solid #3d4f7c;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        text-align: center;
        margin: 20px 0;
    }

    .bearish-box  { border-color: #e74c3c; box-shadow: 0 8px 32px rgba(231,76,60,0.3);  }
    .bullish-box  { border-color: #2ecc71; box-shadow: 0 8px 32px rgba(46,204,113,0.3); }
    .neutral-box  { border-color: #3498db; box-shadow: 0 8px 32px rgba(52,152,219,0.3); }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7, #00d2ff);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1rem;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(123,47,247,0.4);
    }

    /* Text area */
    .stTextArea textarea {
        background: #1e2a4a;
        color: white;
        border: 2px solid #3d4f7c;
        border-radius: 10px;
        font-size: 1rem;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #1e2a4a;
        color: white;
        border: 2px solid #3d4f7c;
        border-radius: 10px;
    }

    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #7b2ff7, #00d2ff, transparent);
        margin: 20px 0;
        border: none;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a, #1a1a2e);
        border-right: 1px solid #3d4f7c;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
LABEL_MAP   = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
LABEL_EMOJI = {0: '🐻',      1: '🐂',      2: '😐'}
LABEL_COLOR = {0: '#e74c3c', 1: '#2ecc71', 2: '#3498db'}
LABEL_CLASS = {0: 'bearish-box', 1: 'bullish-box', 2: 'neutral-box'}
LABEL_DESC  = {
    0: 'Negative / Pessimistic outlook detected',
    1: 'Positive / Optimistic outlook detected',
    2: 'Objective / Mixed sentiment detected'
}
MAX_LEN = 20

# ── Cleaning Function ─────────────────────────────────────────────
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\$([a-zA-Z]+)', r'\1', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Encode Text ───────────────────────────────────────────────────
def encode_text(text, vocab, max_len=MAX_LEN):
    tokens  = text.split()[:max_len]
    indices = [vocab.get(tok, 1) for tok in tokens]
    indices += [0] * (max_len - len(indices))
    return indices

# ── Model Definitions ─────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx, dropout=0.3):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded  = self.dropout(self.embedding(x))
        _, hidden = self.rnn(embedded)
        hidden    = hidden.squeeze(0)
        return self.fc(self.dropout(hidden))


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True, dropout=dropout)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded       = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden         = hidden[-1]
        return self.fc(self.dropout(hidden))


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru       = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded  = self.dropout(self.embedding(x))
        _, hidden = self.gru(embedded)
        hidden    = hidden[-1]
        return self.fc(self.dropout(hidden))


# ── Load Model ────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_choice):
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("model_config.pkl", "rb") as f:
        config = pickle.load(f)

    vs = config["vocab_size"]
    ed = config["embed_dim"]
    hd = config["hidden_dim"]
    od = config["output_dim"]
    pi = config["pad_idx"]

    if model_choice == "Simple RNN":
        model = SimpleRNN(vs, ed, hd, od, pi)
        model.load_state_dict(torch.load("rnn_model.pt",  map_location="cpu"))
    elif model_choice == "LSTM":
        model = LSTMModel(vs, ed, hd, od, pi)
        model.load_state_dict(torch.load("lstm_model.pt", map_location="cpu"))
    else:
        model = GRUModel(vs, ed, hd, od, pi)
        model.load_state_dict(torch.load("gru_model.pt",  map_location="cpu"))

    model.eval()
    return model, vocab, config


# ── Predict ───────────────────────────────────────────────────────
def predict(text, model, vocab):
    cleaned  = clean_tweet(text)
    encoded  = encode_text(cleaned, vocab)
    tensor   = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs, cleaned


# ══════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📈 Financial Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered sentiment classification for financial tweets & headlines</div>', unsafe_allow_html=True)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Top Stats Row ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="metric-card">
        <h2 style="color:#00d2ff">9,536</h2>
        <p style="color:#a0aec0">Training Samples</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="metric-card">
        <h2 style="color:#7b2ff7">3</h2>
        <p style="color:#a0aec0">Sentiment Classes</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="metric-card">
        <h2 style="color:#2ecc71">3</h2>
        <p style="color:#a0aec0">Trained Models</p>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="metric-card">
        <h2 style="color:#e74c3c">NLP</h2>
        <p style="color:#a0aec0">Deep Learning</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two Column Layout ─────────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("### 🤖 Select Model")
    model_choice = st.selectbox(
        label="",
        options=["Simple RNN", "LSTM", "GRU"],
        index=2
    )

    # Model info badges
    model_info = {
        "Simple RNN": ("Fast & lightweight", "#e74c3c"),
        "LSTM"      : ("Long-term memory gates", "#f39c12"),
        "GRU"       : ("Best for short texts ⭐", "#2ecc71")
    }
    info_text, info_color = model_info[model_choice]
    st.markdown(
        f'<span style="background:{info_color}22; color:{info_color}; '
        f'padding:5px 12px; border-radius:20px; font-size:0.85rem; '
        f'border:1px solid {info_color}">{info_text}</span>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📝 Enter Tweet or Headline")
    user_input = st.text_area(
        label="",
        placeholder="e.g. $AAPL - JPMorgan raises price target on Apple citing strong demand...",
        height=140
    )

    # Example buttons
    st.markdown("**💡 Try an example:**")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("🐻 Bearish"):
        user_input = "$BYND - JPMorgan cuts price target citing weak demand outlook"
    if ex2.button("🐂 Bullish"):
        user_input = "$AAPL - Goldman Sachs upgrades to Buy with strong revenue outlook"
    if ex3.button("😐 Neutral"):
        user_input = "$TSLA - Analysts maintain Hold rating ahead of quarterly earnings"

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")


with right_col:
    st.markdown("### 📊 Class Distribution")

    # Dark themed chart
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    classes = ["Bearish", "Bullish", "Neutral"]
    values  = [1442, 1923, 6178]
    colors  = ['#e74c3c', '#2ecc71', '#3498db']

    bars = ax.bar(classes, values, color=colors, edgecolor='white',
                  linewidth=0.5, alpha=0.9, width=0.5)
    ax.set_ylabel("Count", color='white')
    ax.set_title("Training Data Distribution", color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#3d4f7c')
    ax.spines['left'].set_color('#3d4f7c')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 80,
                f'{val:,}', ha='center', color='white',
                fontweight='bold', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Donut chart
    st.markdown("### 🥧 Class Share")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    fig2.patch.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#1a1a2e')

    wedges, texts, autotexts = ax2.pie(
        values,
        labels=classes,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='#1a1a2e', linewidth=2)
    )
    for text in texts:
        text.set_color('white')
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title("Sentiment Share", color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ── Results Section ───────────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

model, vocab, config = load_model(model_choice)

if analyze_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        pred_idx, probs, cleaned = predict(user_input, model, vocab)
        label = LABEL_MAP[pred_idx]
        emoji = LABEL_EMOJI[pred_idx]
        color = LABEL_COLOR[pred_idx]
        desc  = LABEL_DESC[pred_idx]
        bclass = LABEL_CLASS[pred_idx]

        # Result card
        st.markdown(f"""
        <div class="result-box {bclass}">
            <div style="font-size:4rem">{emoji}</div>
            <div style="font-size:2.5rem; font-weight:900; color:{color}">{label}</div>
            <div style="color:#a0aec0; font-size:1rem; margin-top:8px">{desc}</div>
            <div style="font-size:2rem; font-weight:bold; color:white; margin-top:15px">
                {probs[pred_idx]*100:.1f}% Confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"🧹 **Cleaned Input:** `{cleaned}`")
        st.markdown("<br>", unsafe_allow_html=True)

        # Probability charts side by side
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("#### 📊 Probability Bar Chart")
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            fig3.patch.set_facecolor('#1a1a2e')
            ax3.set_facecolor('#1a1a2e')

            bar_colors = ['#e74c3c', '#2ecc71', '#3498db']
            bar_labels = ['Bearish 🐻', 'Bullish 🐂', 'Neutral 😐']
            bars3 = ax3.barh(bar_labels, probs * 100,
                             color=bar_colors, edgecolor='white',
                             linewidth=0.5, alpha=0.9)
            ax3.set_xlim(0, 115)
            ax3.set_xlabel("Probability (%)", color='white')
            ax3.tick_params(colors='white')
            ax3.spines['bottom'].set_color('#3d4f7c')
            ax3.spines['left'].set_color('#3d4f7c')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)

            for bar, val in zip(bars3, probs):
                ax3.text(bar.get_width() + 2,
                         bar.get_y() + bar.get_height()/2,
                         f"{val*100:.1f}%", va='center',
                         color='white', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with chart_right:
            st.markdown("#### 🥧 Probability Pie Chart")
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            fig4.patch.set_facecolor('#1a1a2e')
            ax4.set_facecolor('#1a1a2e')

            explode = [0.05, 0.05, 0.05]
            explode[pred_idx] = 0.15

            wedges, texts, autotexts = ax4.pie(
                probs,
                labels=['Bearish', 'Bullish', 'Neutral'],
                colors=['#e74c3c', '#2ecc71', '#3498db'],
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                wedgeprops=dict(edgecolor='#1a1a2e', linewidth=2)
            )
            for text in texts:
                text.set_color('white')
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)

            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        # Confidence gauge
        st.markdown("#### ⚡ Confidence Meter")
        conf = probs[pred_idx]
        conf_color = '#2ecc71' if conf > 0.7 else '#f39c12' if conf > 0.4 else '#e74c3c'
        st.markdown(f"""
        <div style="background:#1e2a4a; border-radius:10px; padding:5px; border:1px solid #3d4f7c">
            <div style="background:{conf_color}; width:{conf*100:.0f}%;
                        height:25px; border-radius:8px; transition:all 0.5s;
                        display:flex; align-items:center; justify-content:center;
                        color:white; font-weight:bold; font-size:0.9rem">
                {conf*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="text-align:center; padding:10px">
    <h2 style="color:#00d2ff">📈 FinSenti</h2>
    <p style="color:#a0aec0; font-size:0.85rem">Financial Sentiment AI</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Details")

model_details = {
    "Simple RNN" : {"params": "~2.1M", "layers": "1",  "type": "Vanilla RNN"},
    "LSTM"       : {"params": "~4.2M", "layers": "2",  "type": "Long Short-Term Memory"},
    "GRU"        : {"params": "~3.2M", "layers": "2",  "type": "Gated Recurrent Unit"},
}
details = model_details[model_choice]
st.sidebar.markdown(f"""
- **Type**       : {details['type']}
- **Layers**     : {details['layers']}
- **Parameters** : {details['params']}
- **Vocab Size** : {config['vocab_size']:,}
- **Embed Dim**  : {config['embed_dim']}
- **Hidden Dim** : {config['hidden_dim']}
- **Max Length** : {MAX_LEN} tokens
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Sentiment Guide")
st.sidebar.markdown("""
🐻 **Bearish** — Negative outlook
> Price cuts, downgrades, weak demand

🐂 **Bullish** — Positive outlook
> Upgrades, strong earnings, buy ratings

😐 **Neutral** — Mixed/Objective
> Hold ratings, analyst coverage
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Dataset Info")
st.sidebar.markdown("""
- **Source** : Twitter Financial News
- **Train**  : 9,536 tweets
- **Valid**  : 2,388 tweets
- **Lang**   : English
- **Domain** : Finance & FinTech
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="text-align:center; color:#a0aec0; font-size:0.8rem">'
    'Built with PyTorch & Streamlit</p>',
    unsafe_allow_html=True
)