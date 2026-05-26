import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    CNN pour la reconnaissance de chiffres dans les cellules Sudoku.
    Entraîné sur MNIST + données augmentées.
    Input : (B, 1, 28, 28) images en niveaux de gris
    Output : (B, 10) logits — classes 0-9 (0 = cellule vide)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),               # 28→14
            nn.Dropout2d(0.1),
            # Bloc 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),               # 14→7
            nn.Dropout2d(0.1),
            # Bloc 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),       # →2×2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)
