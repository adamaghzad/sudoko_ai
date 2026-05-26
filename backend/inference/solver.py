"""
Moteur d'inférence — charge tous les modèles et résout des puzzles Sudoku.
Fournit aussi un solveur backtracking classique comme baseline.
"""
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from config import DEVICE, WEIGHTS_DIR
from utils.sudoku_utils import is_valid_solution, solve_deterministic, validate_puzzle_input
from models.mlp_solver import MLPSolver
from models.cnn_solver import CNNSolver
from models.rnn_solver import RNNSolver
from models.lstm_solver import LSTMSolver
from models.gru_solver import GRUSolver
from models.hybrid_solver import HybridSolver


MODEL_REGISTRY = {
    "mlp":    (MLPSolver,    {"hidden_sizes": [1024, 512, 256, 128], "dropout": 0.3}),
    "cnn":    (CNNSolver,    {"base_channels": 64, "num_blocks": 6, "dropout": 0.2}),
    "rnn":    (RNNSolver,    {"embed_dim": 16, "hidden_size": 256, "num_layers": 2}),
    "lstm":   (LSTMSolver,   {"embed_dim": 32, "hidden_size": 256, "num_layers": 2}),
    "gru":    (GRUSolver,    {"embed_dim": 32, "hidden_size": 256, "num_layers": 2}),
    "hybrid": (HybridSolver, {"cnn_channels": 64, "hidden_size": 256, "num_lstm_layers": 2}),
}

MODEL_META = {
    "mlp":    {"color": "#6366f1", "arch": "MLP",    "desc": "Perceptron Multicouche — données tabulaires"},
    "cnn":    {"color": "#8b5cf6", "arch": "CNN",    "desc": "Réseau Convolutif — biais inductif spatial"},
    "rnn":    {"color": "#ec4899", "arch": "RNN",    "desc": "Réseau Récurrent — séquence bidirectionnelle"},
    "lstm":   {"color": "#f59e0b", "arch": "LSTM",   "desc": "LSTM — mémoire longue, 3 portes"},
    "gru":    {"color": "#10b981", "arch": "GRU",    "desc": "GRU — 2 portes, compromis LSTM"},
    "hybrid": {"color": "#3b82f6", "arch": "Hybrid", "desc": "CNN + LSTM — spatial + temporel"},
}


def backtrack_solve(puzzle: list[int]) -> Optional[list[int]]:
    """Solveur classique par backtracking — garantit la solution exacte."""
    board = [list(row) for row in np.array(puzzle).reshape(9, 9).tolist()]
    if solve_deterministic(board):
        return [cell for row in board for cell in row]
    return None


class SudokuSolverEngine:
    def __init__(self):
        self.models: dict[str, torch.nn.Module] = {}
        self.trained_models: set[str] = set()
        self._load_all_models()

    def _load_all_models(self):
        """Charge tous les modèles dont les poids sont disponibles."""
        for name, (cls, kwargs) in MODEL_REGISTRY.items():
            weight_path = Path(WEIGHTS_DIR) / f"{name}.pt"
            model = cls(**kwargs)
            if weight_path.exists():
                try:
                    state = torch.load(weight_path, map_location=DEVICE)
                    model.load_state_dict(state)
                    model.eval()
                    model.to(DEVICE)
                    self.trained_models.add(name)
                    print(f"Modèle chargé: {name}")
                except Exception as e:
                    print(f"Erreur chargement {name}: {e}")
            self.models[name] = model

    def _nn_solve(self, model: torch.nn.Module, puzzle: list[int]) -> tuple:
        """Inférence réseau de neurones — retourne (solution, conf_mean, conf_empty, conf_min, conf_max)."""
        with torch.no_grad():
            x = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(x)                            # (1, 81, 9)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).squeeze(0) + 1  # 1-9
            cell_conf = probs.max(dim=-1).values.squeeze(0)  # (81,)

            empty_mask = torch.tensor([v == 0 for v in puzzle], dtype=torch.bool)
            conf_mean  = cell_conf.mean().item()
            conf_empty = cell_conf[empty_mask].mean().item() if empty_mask.any() else conf_mean
            conf_min   = cell_conf.min().item()
            conf_max   = cell_conf.max().item()

        solution = preds.cpu().tolist()
        for i, v in enumerate(puzzle):
            if v != 0:
                solution[i] = v
        return solution, conf_mean, conf_empty, conf_min, conf_max

    def solve(self, puzzle: list[int]) -> dict:
        """
        Résout un puzzle avec tous les modèles disponibles + backtracking.
        Retourne un dict structuré avec les résultats de chaque modèle.
        """
        if len(puzzle) != 81:
            raise ValueError("Le puzzle doit contenir exactement 81 valeurs")

        puzzle_valid, puzzle_error = validate_puzzle_input(puzzle)
        puzzle_arr = np.array(puzzle)
        results = {}

        # 1. Backtracking (toujours disponible)
        t0 = time.perf_counter()
        bt_solution = backtrack_solve(puzzle)
        bt_time = (time.perf_counter() - t0) * 1000
        bt_conf = 1.0 if bt_solution else 0.0
        results["backtracking"] = {
            "solution": bt_solution,
            "time_ms": round(bt_time, 2),
            "is_valid": is_valid_solution(puzzle_arr, np.array(bt_solution)) if bt_solution else False,
            "confidence": bt_conf,
            "confidence_empty": bt_conf,
            "confidence_min": bt_conf,
            "confidence_max": bt_conf,
            "trained": True,
            "color": "#64748b",
            "arch": "Backtracking",
            "desc": "Algorithme classique — garantit la solution exacte",
        }

        # 2. Modèles de deep learning
        for name, model in self.models.items():
            meta = MODEL_META[name]
            is_trained = name in self.trained_models

            t0 = time.perf_counter()
            if is_trained:
                try:
                    solution, conf_mean, conf_empty, conf_min, conf_max = self._nn_solve(model, puzzle)
                    sol_arr = np.array(solution)
                    valid = is_valid_solution(puzzle_arr, sol_arr)
                except Exception:
                    solution, conf_mean, conf_empty, conf_min, conf_max, valid = None, 0.0, 0.0, 0.0, 0.0, False
            else:
                solution, conf_mean, conf_empty, conf_min, conf_max, valid = None, 0.0, 0.0, 0.0, 0.0, False

            elapsed = (time.perf_counter() - t0) * 1000
            results[name] = {
                "solution": solution,
                "time_ms": round(elapsed, 2),
                "is_valid": valid,
                "confidence": round(conf_mean, 4),
                "confidence_empty": round(conf_empty, 4),
                "confidence_min": round(conf_min, 4),
                "confidence_max": round(conf_max, 4),
                "trained": is_trained,
                **meta,
            }

        return {
            "puzzle": puzzle,
            "results": results,
            "num_clues": sum(1 for v in puzzle if v != 0),
            "puzzle_valid": puzzle_valid,
            "puzzle_error": puzzle_error,
        }

    def get_status(self) -> dict:
        """Retourne le statut de tous les modèles."""
        status = {}
        for name in MODEL_REGISTRY:
            weight_path = Path(WEIGHTS_DIR) / f"{name}.pt"
            hist_path = Path(WEIGHTS_DIR) / f"{name}_history.json"
            history = None
            if hist_path.exists():
                with open(hist_path) as f:
                    history = json.load(f)
            status[name] = {
                "trained": name in self.trained_models,
                "weight_exists": weight_path.exists(),
                "weight_size_mb": round(weight_path.stat().st_size / 1e6, 2)
                    if weight_path.exists() else 0,
                "history": history,
                **MODEL_META[name],
            }
        return status
