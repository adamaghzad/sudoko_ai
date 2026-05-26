import torch
import torch.nn as nn


class GRUSolver(nn.Module):
    """
    Gated Recurrent Unit (GRU).
    Alternative légère au LSTM : 2 portes (reset, update), 1 seul état caché.
    Meilleur compromis vitesse / performance que le LSTM.
    Input : (B, 81) int64
    Output : (B, 81, 9) logits
    """

    def __init__(self, embed_dim: int = 32, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(10, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 81)
        emb = self.embedding(x.long())         # (B, 81, embed_dim)
        out, _ = self.gru(emb)                 # (B, 81, hidden*2)
        out = self.ln(self.dropout(out))
        return self.fc(out)                    # (B, 81, 9)

    @property
    def name(self) -> str:
        return "GRU"

    @property
    def description(self) -> str:
        return "GRU — 2 portes, plus léger que LSTM, bidirectionnel"
