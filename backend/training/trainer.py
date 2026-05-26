import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import DEVICE, WEIGHTS_DIR, GRAD_CLIP


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor,
                    puzzles: torch.Tensor) -> dict:
    """
    Cell accuracy (toutes cellules) + puzzle accuracy (grille entière correcte).
    On évalue seulement les cellules vides (puzzle==0) pour puzzle_acc.
    """
    preds = logits.argmax(dim=-1) + 1          # 1-9
    correct_cells = (preds == targets).float()

    # Puzzle accuracy : toutes les cellules correctes
    puzzle_correct = correct_cells.all(dim=-1).float()

    # Cell accuracy sur cellules vides uniquement
    empty_mask = (puzzles == 0)
    empty_correct = (correct_cells * empty_mask).sum() / empty_mask.sum().clamp(min=1)

    return {
        "cell_acc": correct_cells.mean().item(),
        "empty_cell_acc": empty_correct.item(),
        "puzzle_acc": puzzle_correct.mean().item(),
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4,
                patience: int = 8, grad_clip: float = GRAD_CLIP,
                use_recurrent_clip: bool = False) -> dict:
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "train_cell_acc": [],
               "val_cell_acc": [], "val_puzzle_acc": [], "val_empty_cell_acc": [],
               "epochs_trained": 0}

    best_val_loss = float("inf")
    best_val_cell_acc = 0.0
    best_epoch = 1
    best_state = None
    model_name = model.name.lower()
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"Training: {model.name} | Device: {DEVICE}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # ── Train ──
        model.train()
        train_loss, train_metrics = 0.0, {"cell_acc": 0.0}
        for puzzles, solutions in train_loader:
            puzzles, solutions = puzzles.to(DEVICE), solutions.to(DEVICE)
            optimizer.zero_grad()
            logits = model(puzzles)          # (B, 81, 9)
            # targets: solutions are 1-9, convert to 0-8 for CrossEntropy
            targets = (solutions - 1).long()  # (B, 81)
            loss = criterion(logits.view(-1, 9), targets.view(-1))
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
            m = compute_metrics(logits.detach(), solutions, puzzles)
            train_metrics["cell_acc"] += m["cell_acc"]

        n = len(train_loader)
        train_loss /= n
        train_metrics = {k: v / n for k, v in train_metrics.items()}

        # ── Validate ──
        model.eval()
        val_loss, val_metrics = 0.0, {"cell_acc": 0.0, "puzzle_acc": 0.0, "empty_cell_acc": 0.0}
        with torch.no_grad():
            for puzzles, solutions in val_loader:
                puzzles, solutions = puzzles.to(DEVICE), solutions.to(DEVICE)
                logits = model(puzzles)
                targets = (solutions - 1).long()
                loss = criterion(logits.view(-1, 9), targets.view(-1))
                val_loss += loss.item()
                m = compute_metrics(logits, solutions, puzzles)
                for k in val_metrics:
                    val_metrics[k] += m[k]

        n = len(val_loader)
        val_loss /= n
        val_metrics = {k: v / n for k, v in val_metrics.items()}

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cell_acc"].append(train_metrics["cell_acc"])
        history["val_cell_acc"].append(val_metrics["cell_acc"])
        history["val_puzzle_acc"].append(val_metrics["puzzle_acc"])
        history["val_empty_cell_acc"].append(val_metrics["empty_cell_acc"])
        history["epochs_trained"] = epoch

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss {train_loss:.4f}/{val_loss:.4f} | "
              f"CellAcc {train_metrics['cell_acc']:.3f}/{val_metrics['cell_acc']:.3f} | "
              f"PuzzleAcc {val_metrics['puzzle_acc']:.3f} | "
              f"{elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_cell_acc = val_metrics["cell_acc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if stopper.step(val_loss):
            print(f"Early stopping à l'époque {epoch}")
            break

    # Save best weights
    weight_path = Path(WEIGHTS_DIR) / f"{model_name}.pt"
    torch.save(best_state, weight_path)
    print(f"Poids sauvegardés: {weight_path}")

    # Save history
    hist_path = Path(WEIGHTS_DIR) / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    history["best_val_puzzle_acc"] = max(history["val_puzzle_acc"])
    history["best_val_cell_acc"] = best_val_cell_acc
    history["best_epoch"] = best_epoch
    history["train_time_sec"] = round(time.time() - total_start, 1)
    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> dict:
    model = model.to(DEVICE)
    model.eval()
    metrics = {"cell_acc": 0.0, "puzzle_acc": 0.0, "empty_cell_acc": 0.0}
    n = 0
    with torch.no_grad():
        for puzzles, solutions in test_loader:
            puzzles, solutions = puzzles.to(DEVICE), solutions.to(DEVICE)
            logits = model(puzzles)
            m = compute_metrics(logits, solutions, puzzles)
            for k in metrics:
                metrics[k] += m[k]
            n += 1
    return {k: v / n for k, v in metrics.items()}
