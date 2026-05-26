import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSolver(nn.Module):
    """
    Architecture Hybride CNN + LSTM.
    CNN : extrait les features spatiales de chaque ligne de la grille.
    LSTM : capture les dépendances entre lignes (temporalité inter-rangées).
    Combine biais inductif spatial (CNN) et temporel (LSTM).
    Input : (B, 81) int64
    Output : (B, 81, 9) logits
    """

    def __init__(self, cnn_channels: int = 64, hidden_size: int = 256,
                 num_lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # CNN global : features spatiales sur la grille complète
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(10, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            nn.Conv2d(cnn_channels * 2, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
        )
        # After CNN: (B, cnn_channels, 9, 9)
        # Process row-by-row with LSTM: sequence = rows, features per step = cnn_channels * 9
        lstm_input_size = cnn_channels * 9
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size * 2)
        # Output: for each of the 9 rows, predict 9 cells × 9 digits
        self.fc = nn.Linear(hidden_size * 2, 9 * 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_oh = F.one_hot(x.long(), num_classes=10).float()    # (B, 81, 10)
        x_grid = x_oh.view(B, 9, 9, 10).permute(0, 3, 1, 2)  # (B, 10, 9, 9)

        cnn_out = self.spatial_cnn(x_grid)                    # (B, ch, 9, 9)
        # Reshape to (B, 9_rows, ch*9_cols) — treat rows as sequence
        seq = cnn_out.permute(0, 2, 1, 3).contiguous()        # (B, 9, ch, 9)
        seq = seq.view(B, 9, -1)                               # (B, 9, ch*9)

        lstm_out, _ = self.lstm(seq)                           # (B, 9, hidden*2)
        lstm_out = self.ln(self.dropout(lstm_out))

        out = self.fc(lstm_out)                                # (B, 9, 81)
        out = out.view(B, 9, 9, 9)                             # (B, rows, cols, digits)
        return out.view(B, 81, 9)                              # (B, 81, 9)

    @property
    def name(self) -> str:
        return "Hybrid"

    @property
    def description(self) -> str:
        return "CNN + LSTM — features spatiales (CNN) + dépendances temporelles (LSTM)"
