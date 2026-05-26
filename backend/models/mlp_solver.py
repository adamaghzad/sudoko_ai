import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPSolver(nn.Module):
    """
    Perceptron Multicouche (MLP) — données tabulaires.
    Input : grille 9×9 aplatie (81 valeurs 0-9, 0=vide)
    Output : logits par cellule (81 × 9)
    """

    def __init__(self, hidden_sizes: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256, 128]

        # One-hot encoding: 81 cells × 10 classes = 810 features
        layers = []
        in_dim = 81 * 10
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 81 * 9))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 81) int64 values 0-9
        x_oh = F.one_hot(x.long(), num_classes=10).float()  # (B, 81, 10)
        x_flat = x_oh.view(x.size(0), -1)                   # (B, 810)
        out = self.net(x_flat)                               # (B, 729)
        return out.view(-1, 81, 9)                           # (B, 81, 9)

    @property
    def name(self) -> str:
        return "MLP"

    @property
    def description(self) -> str:
        return "Perceptron Multicouche — données tabulaires, fully-connected"
