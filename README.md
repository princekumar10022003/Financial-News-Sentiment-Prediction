# Financial-News-Sentiment-Prediction
Project Assigned From Guvi HCL
---

## 📊 Pipeline Steps

### Phase 1 — Data Preprocessing
- Load raw CSV files
- Clean tweets: remove URLs, mentions, hashtags
- Keep financial tickers (strip $ sign)
- Remove punctuation and numbers
- Collapse whitespace
- Drop empty rows after cleaning

### Phase 2 — Vocabulary & Dataset
- Build vocabulary from training data only (no leakage)
- Minimum frequency threshold = 2
- Special tokens: `<PAD>` (index 0), `<UNK>` (index 1)
- Fixed sequence length: MAX_LEN = 20
- PyTorch Dataset and DataLoader setup
- Class weights computed for imbalance handling

### Phase 3 — Simple RNN
- Architecture: Embedding → RNN → Dropout → FC
- Embedding dim: 128
- Hidden dim: 256
- Dropout: 0.3
- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss with class weights

### Phase 4 — LSTM
- Architecture: Embedding → LSTM (2 layers) → Dropout → FC
- Same hyperparameters as RNN
- Stacked LSTM with dropout between layers
- Uses last hidden state for classification

### Phase 5 — GRU
- Architecture: Embedding → GRU (2 layers) → Dropout → FC
- Fewer parameters than LSTM
- Better suited for short texts like tweets

### Phase 6 — Model Comparison
- Side by side accuracy and Macro F1 charts
- Class-wise F1 comparison
- All 3 confusion matrices
- Training curve plots

### Phase 7 — FinBERT (Optional)
- Model: ProsusAI/finbert
- Lightweight fine-tuning on 15% of training data
- 1 epoch with linear warmup scheduler
- Compared against RNN baseline

### Phase 8 — Streamlit Dashboard
- Dark themed interactive UI
- Select between Simple RNN, LSTM, GRU
- Live sentiment prediction with confidence score
- Probability bar chart and pie chart
- Confidence meter with color coding

---

## 📈 Model Architecture Summary

| Model      | Layers | Parameters | Best For          |
|------------|--------|------------|-------------------|
| Simple RNN | 1      | ~2.1M      | Fast baseline     |
| LSTM       | 2      | ~4.2M      | Long dependencies |
| GRU        | 2      | ~3.2M      | Short texts ⭐    |
| FinBERT    | 12     | ~110M      | Best accuracy     |

---

## 📉 Evaluation Metrics

| Metric        | Description                              |
|---------------|------------------------------------------|
| Accuracy      | Overall correct predictions              |
| Macro F1      | Equal weight to all 3 classes            |
| Precision     | Class-wise precision                     |
| Recall        | Class-wise recall                        |
| Confusion Matrix | Visual error analysis                 |

> **Primary metric: Macro F1** — used because dataset is imbalanced (Neutral = 65%)

---

## 🔑 Key Design Decisions

| Decision | Reason |
|---|---|
| Vocab built on train only | Prevents data leakage |
| Class weights in loss | Handles 65% Neutral imbalance |
| Gradient clipping (max=1.0) | Prevents exploding gradients in RNN |
| MAX_LEN = 20 | Covers 95th percentile of tweet lengths |
| MIN_FREQ = 2 | Removes noise from rare words |
| Macro F1 as primary metric | Equal importance to minority classes |

---

## 🖥️ Streamlit Dashboard Features

- 🌑 Dark gradient theme
- 📊 Dataset distribution charts
- 🤖 Switch between 3 trained models
- 🔍 Live sentiment prediction
- 📈 Probability bar and pie charts
- ⚡ Color-coded confidence meter
- 💡 Example tweets to test quickly

---

## 📦 Dependencies
