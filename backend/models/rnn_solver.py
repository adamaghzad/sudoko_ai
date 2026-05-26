import torch
import torch.nn as nn


class RNNSolver(nn.Module):
    """
    Réseau Récurrent Simple (RNN).
    Traite les 81 cellules comme une séquence temporelle.
    Adapté aux séquences courtes; souffre de la disparition du gradient.
    Input : (B, 81) int64
    Output : (B, 81, 9) logits
    """

    def __init__(self, embed_dim: int = 16, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(10, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 81)
        emb = self.embedding(x.long())         # (B, 81, embed_dim)
        out, _ = self.rnn(emb)                 # (B, 81, hidden*2)
        out = self.dropout(out)
        return self.fc(out)                    # (B, 81, 9)

    @property
    def name(self) -> str:
        return "RNN"

    @property
    def description(self) -> str:
        return "Réseau Récurrent — séquence de 81 cellules, bidirectionnel"
