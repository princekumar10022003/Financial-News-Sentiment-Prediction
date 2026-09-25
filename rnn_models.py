"""
rnn_models.py
-------------
Three RNN-based sentiment classifiers:
  - SimpleRNNClassifier
  - LSTMClassifier
  - GRUClassifier

All share the same Embedding + Dropout + Linear head.
The only difference is the recurrent layer inside.
"""

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    Generic recurrent classifier.
    Supports rnn_type = 'rnn' | 'lstm' | 'gru'.

    Architecture:
        Embedding(vocab_size, embed_dim)
        → RNN/LSTM/GRU(embed_dim, hidden_dim, num_layers, bidirectional)
        → Dropout
        → Linear(hidden_dim * directions, num_classes)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.rnn_type     = rnn_type.lower()
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.bidirectional = bidirectional
        self.directions   = 2 if bidirectional else 1

        # ── Embedding layer ──────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # ── Recurrent layer ──────────────────────────────────────────────
        rnn_kwargs = dict(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(**rnn_kwargs)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError(f"Unknown rnn_type '{rnn_type}'. Choose rnn / lstm / gru.")

        # ── Classifier head ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * self.directions, num_classes)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len)  — integer token ids
        returns logits : (batch, num_classes)
        """
        # Embedding → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # Recurrent layer
        if self.rnn_type == "lstm":
            output, (h_n, _) = self.rnn(embedded)
        else:
            output, h_n = self.rnn(embedded)

        # h_n shape: (num_layers * directions, batch, hidden_dim)
        # Grab the last layer's hidden state for every direction
        if self.bidirectional:
            # Concatenate forward & backward of the last layer
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)   # (batch, hidden*2)
        else:
            h_last = h_n[-1]                                  # (batch, hidden)

        out = self.dropout(h_last)
        logits = self.fc(out)   # (batch, num_classes)
        return logits

    # ─────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"RNNClassifier(type={self.rnn_type}, "
            f"hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, "
            f"bidir={self.bidirectional}, "
            f"params={self.count_parameters():,})"
        )


# ─────────────────────────────────────────────
# Convenience factory functions
# ─────────────────────────────────────────────
def SimpleRNNClassifier(vocab_size, **kwargs):
    """Embedding + vanilla RNN + Linear head."""
    return RNNClassifier(vocab_size, rnn_type="rnn", **kwargs)


def LSTMClassifier(vocab_size, **kwargs):
    """Embedding + LSTM + Linear head."""
    return RNNClassifier(vocab_size, rnn_type="lstm", **kwargs)


def GRUClassifier(vocab_size, **kwargs):
    """Embedding + GRU + Linear head."""
    return RNNClassifier(vocab_size, rnn_type="gru", **kwargs)


# ─────────────────────────────────────────────
# Quick architecture test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    VOCAB = 15002   # PAD + UNK + 15000 words
    BATCH = 8
    SEQ   = 64

    dummy_input = torch.randint(0, VOCAB, (BATCH, SEQ))

    for name, Model in [
        ("SimpleRNN", SimpleRNNClassifier),
        ("LSTM",      LSTMClassifier),
        ("GRU",       GRUClassifier),
    ]:
        model = Model(VOCAB)
        out   = model(dummy_input)
        print(f"{name:10s}  output={out.shape}  {model}")
