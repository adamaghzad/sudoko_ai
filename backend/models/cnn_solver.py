import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class CNNSolver(nn.Module):
    """
    Réseau de Neurones Convolutifs (CNN).
    Traite la grille 9×9 comme une image avec biais inductif spatial.
    Input : (B, 81) → reshape (B, 10, 9, 9) via one-hot
    Output : (B, 81, 9) logits
    """

    def __init__(self, base_channels: int = 64, num_blocks: int = 6, dropout: float = 0.2):
        super().__init__()
        layers = [ConvBlock(10, base_channels)]
        ch = base_channels
        for i in range(1, num_blocks):
            next_ch = min(ch * 2, 512)
            layers.append(ConvBlock(ch, next_ch))
            ch = next_ch
        layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(ch, 9, kernel_size=1))  # pixel-wise classification
        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 81)
        x_oh = F.one_hot(x.long(), num_classes=10).float()   # (B, 81, 10)
        x_grid = x_oh.view(-1, 9, 9, 10).permute(0, 3, 1, 2) # (B, 10, 9, 9)
        out = self.cnn(x_grid)                                 # (B, 9, 9, 9)
        out = out.permute(0, 2, 3, 1).contiguous()             # (B, 9, 9, 9)
        return out.view(-1, 81, 9)                             # (B, 81, 9)

    @property
    def name(self) -> str:
        return "CNN"

    @property
    def description(self) -> str:
        return "Réseau Convolutif — exploite la structure spatiale 9×9 de la grille"
