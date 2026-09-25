"""
preprocessing.py
----------------
Text cleaning, vocabulary building, tokenization, and DataLoader creation
for the Financial News Sentiment Prediction project.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_LEN    = 64       # max tokens per tweet
VOCAB_SIZE = 15000    # top N words kept
BATCH_SIZE = 64
PAD_IDX    = 0
UNK_IDX    = 1

LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}


# ─────────────────────────────────────────────
# 1. Text cleaning
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean a raw financial tweet:
      - Remove URLs
      - Remove @mentions
      - Keep $TICKERS (important sentiment signal)
      - Remove special chars except $ and alphanumerics
      - Collapse whitespace
      - Lowercase
    """
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                       # remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                  # strip # but keep word
    text = re.sub(r"[^a-zA-Z0-9\$\s\.\,\!\?]", " ", text) # keep letters/digits/$
    text = re.sub(r"\s+", " ", text).strip()               # collapse spaces
    text = text.lower()
    return text


# ─────────────────────────────────────────────
# 2. Vocabulary builder (fit on TRAIN only)
# ─────────────────────────────────────────────
class Vocabulary:
    """Maps words to integer indices."""

    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        self.idx2word = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}

    def build(self, texts: list[str]):
        """Build vocabulary from a list of cleaned texts."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Keep top (max_size - 2) words (we already have PAD and UNK)
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary built: {len(self.word2idx):,} tokens")

    def encode(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        """Convert text to a padded/truncated list of indices."""
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        # Pad to max_len
        ids += [PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────
# 3. Data loading pipeline
# ─────────────────────────────────────────────
def load_data(train_path: str, valid_path: str):
    """Load CSVs and return cleaned DataFrames."""
    train_df = pd.read_csv(train_path)
    valid_df  = pd.read_csv(valid_path)

    # Clean text
    train_df["clean"] = train_df["text"].apply(clean_text)
    valid_df["clean"]  = valid_df["text"].apply(clean_text)

    print(f"Train: {len(train_df):,} samples")
    print(f"Valid: {len(valid_df):,}  samples")
    print("\nTrain label distribution:")
    print(train_df["label"].map(LABEL_MAP).value_counts())

    return train_df, valid_df


def build_loaders(train_df: pd.DataFrame,
                  valid_df: pd.DataFrame,
                  vocab: Vocabulary,
                  batch_size: int = BATCH_SIZE):
    """
    Encode texts → tensors → TensorDataset → DataLoader.
    Returns (train_loader, valid_loader).
    """
    def encode_df(df):
        X = np.array([vocab.encode(t) for t in df["clean"]], dtype=np.int64)
        y = np.array(df["label"].values, dtype=np.int64)
        return torch.tensor(X), torch.tensor(y)

    X_train, y_train = encode_df(train_df)
    X_valid, y_valid = encode_df(valid_df)

    train_ds = TensorDataset(X_train, y_train)
    valid_ds  = TensorDataset(X_valid, y_valid)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader  = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Valid batches : {len(valid_loader)}")
    return train_loader, valid_loader


# ─────────────────────────────────────────────
# 4. Class weights (handle imbalance)
# ─────────────────────────────────────────────
def get_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights and return as a CUDA/CPU tensor.
    Pass this to nn.CrossEntropyLoss(weight=...) to handle class imbalance.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=labels
    )
    tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class weights → Bearish:{tensor[0]:.3f}  Bullish:{tensor[1]:.3f}  Neutral:{tensor[2]:.3f}")
    return tensor


# ─────────────────────────────────────────────
# 5. Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_df, valid_df = load_data("data/sent_train.csv", "data/sent_valid.csv")

    vocab = Vocabulary()
    vocab.build(train_df["clean"].tolist())

    train_loader, valid_loader = build_loaders(train_df, valid_df, vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = get_class_weights(train_df["label"].values, device)

    # Show a batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shape — X: {X_batch.shape}, y: {y_batch.shape}")
    print("Labels in batch:", y_batch[:8].tolist())
