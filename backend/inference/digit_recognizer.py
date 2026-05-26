"""
Reconnaissance de chiffres dans les cellules Sudoku extraites.
Utilise le DigitCNN entraîné sur MNIST ou charge des poids pré-entraînés.
"""
import numpy as np
import torch
from pathlib import Path

from models.digit_cnn import DigitCNN
from config import DEVICE, WEIGHTS_DIR, CELL_SIZE


class DigitRecognizer:
    def __init__(self):
        self.model = DigitCNN().to(DEVICE)
        weight_path = Path(WEIGHTS_DIR) / "digit_cnn.pt"

        if weight_path.exists():
            state = torch.load(weight_path, map_location=DEVICE)
            self.model.load_state_dict(state)
            print(f"DigitCNN chargé depuis {weight_path}")
        else:
            print("[AVERTISSEMENT] Poids DigitCNN non trouvés — utilisation MNIST intégré")
            self._load_mnist_pretrained()

        self.model.eval()

    def _load_mnist_pretrained(self):
        """
        Entraîne rapidement le DigitCNN sur MNIST si les poids ne sont pas disponibles.
        Requiert torchvision.
        """
        try:
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            from torch.optim import Adam

            transform = transforms.Compose([
                transforms.Resize((CELL_SIZE, CELL_SIZE)),
                transforms.ToTensor(),
            ])
            train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
            loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = Adam(self.model.parameters(), lr=1e-3)

            print("Entraînement DigitCNN sur MNIST (5 époques)...")
            self.model.train()
            for epoch in range(5):
                total_loss = 0.0
                for imgs, labels in loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(self.model(imgs), labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"  Époque {epoch+1}/5 — Loss: {total_loss/len(loader):.4f}")

            torch.save(self.model.state_dict(), Path(WEIGHTS_DIR) / "digit_cnn.pt")
            self.model.eval()
            print("DigitCNN entraîné et sauvegardé.")
        except Exception as e:
            print(f"Impossible d'entraîner DigitCNN: {e}")

    @torch.no_grad()
    def predict_cells(self, cells: np.ndarray, empty_flags: list[bool]) -> list[int]:
        """
        Prédit le chiffre (0-9) pour chaque cellule.
        cells: (81, 1, 28, 28) float32 ∈ [0,1]
        Retourne une liste de 81 entiers (0 = vide).
        """
        digits = []
        tensor = torch.tensor(cells, dtype=torch.float32).to(DEVICE)  # (81, 1, 28, 28)
        logits = self.model(tensor)                                     # (81, 10)
        preds = logits.argmax(dim=1).cpu().numpy()                      # (81,)

        for i, (pred, is_empty) in enumerate(zip(preds, empty_flags)):
            if is_empty:
                digits.append(0)
            else:
                digits.append(int(pred))

        return digits

    @torch.no_grad()
    def predict_with_confidence(self, cells: np.ndarray,
                                 empty_flags: list[bool]) -> list[dict]:
        """Retourne les prédictions avec scores de confiance."""
        tensor = torch.tensor(cells, dtype=torch.float32).to(DEVICE)
        probs = torch.softmax(self.model(tensor), dim=1).cpu().numpy()  # (81, 10)

        results = []
        for i, (prob, is_empty) in enumerate(zip(probs, empty_flags)):
            if is_empty:
                results.append({"digit": 0, "confidence": 1.0, "is_empty": True})
            else:
                digit = int(prob.argmax())
                results.append({
                    "digit": digit,
                    "confidence": float(prob[digit]),
                    "is_empty": False,
                })
        return results
