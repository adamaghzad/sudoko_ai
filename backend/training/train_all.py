"""
Entraînement de tous les modèles Sudoku AI.

Usage :
  python -m training.train_all                           # pipeline complet
  python -m training.train_all --quick                   # test rapide (5 époques)
  python -m training.train_all --kaggle data/sudoku.csv  # dataset Kaggle
  python -m training.train_all --model lstm              # un seul modèle
  python -m training.train_all --skip-existing           # ne ré-entraîne pas si poids OK
"""
import sys, json, random, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                    PATIENCE, WEIGHTS_DIR)
from training.dataset import build_loaders
from training.trainer import train_model, evaluate_model
from models.mlp_solver    import MLPSolver
from models.cnn_solver    import CNNSolver
from models.rnn_solver    import RNNSolver
from models.lstm_solver   import LSTMSolver
from models.gru_solver    import GRUSolver
from models.hybrid_solver import HybridSolver


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_all_models() -> list:
    return [
        MLPSolver(hidden_sizes=[1024, 512, 256, 128], dropout=0.3),
        CNNSolver(base_channels=64, num_blocks=6, dropout=0.2),
        RNNSolver(embed_dim=16, hidden_size=256, num_layers=2, dropout=0.3),
        LSTMSolver(embed_dim=32, hidden_size=256, num_layers=2, dropout=0.3),
        GRUSolver(embed_dim=32, hidden_size=256, num_layers=2, dropout=0.3),
        HybridSolver(cnn_channels=64, hidden_size=256, num_lstm_layers=2, dropout=0.3),
    ]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--quick",         action="store_true", help="5 époques, petit dataset")
    pa.add_argument("--model",         type=str, default=None,
                    help="mlp | cnn | rnn | lstm | gru | hybrid")
    pa.add_argument("--epochs",        type=int, default=EPOCHS)
    pa.add_argument("--kaggle",        type=str, default=None, metavar="CSV",
                    help="Chemin vers data/sudoku.csv")
    pa.add_argument("--skip-existing", action="store_true",
                    help="Saute les modèles dont les poids existent déjà")
    pa.add_argument("--data-dir",      type=str, default="data")
    args = pa.parse_args()

    set_seed(42)
    epochs = 5 if args.quick else args.epochs

    print(f"\n{'='*60}")
    print(f"  Sudoku AI — Entraînement")
    print(f"  Mode    : {'RAPIDE' if args.quick else 'COMPLET'}")
    print(f"  Époques : {epochs}")
    print(f"  Device  : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"{'='*60}\n")

    # DataLoaders — le pipeline (data_pipeline.py) se lance automatiquement si besoin
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=BATCH_SIZE,
        kaggle_csv=args.kaggle,
        data_dir=args.data_dir,
        max_train=10_000 if args.quick else None,
        max_val=1_000   if args.quick else None,
        max_test=500    if args.quick else None,
    )

    models = build_all_models()
    if args.model:
        models = [m for m in models if m.name.lower() == args.model.lower()]
        if not models:
            print(f"Modèle inconnu : {args.model}. Choix : mlp cnn rnn lstm gru hybrid")
            sys.exit(1)

    all_results = {}
    weights_dir = Path(WEIGHTS_DIR)

    for model in models:
        name = model.name.lower()
        weight_path = weights_dir / f"{name}.pt"

        if args.skip_existing and weight_path.exists():
            print(f"\n[SKIP] {model.name} — poids déjà présents ({weight_path})")
            continue

        print(f"\n{'─'*60}")
        print(f"  Modèle : {model.name}")
        print(f"{'─'*60}")

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
        )
        test_metrics = evaluate_model(model, test_loader)

        print(f"\n  [TEST] {model.name}")
        for k, v in test_metrics.items():
            print(f"    {k:<20} : {v:.4f}")

        all_results[name] = {"history": history, "test_metrics": test_metrics}

    if all_results:
        summary_path = weights_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  {'Modèle':<10} {'CellAcc':>10} {'PuzzleAcc':>12} {'EmptyAcc':>12}")
        print(f"  {'-'*56}")
        for name, res in all_results.items():
            m = res["test_metrics"]
            print(f"  {name:<10} {m['cell_acc']:>10.4f} {m['puzzle_acc']:>12.4f} "
                  f"{m.get('empty_cell_acc', 0):>12.4f}")
        print(f"{'='*70}")
        print(f"\n  Résumé : {summary_path}")

    return all_results


if __name__ == "__main__":
    main()
