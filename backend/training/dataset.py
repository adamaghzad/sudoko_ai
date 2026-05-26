"""
Dataset PyTorch — charge les fichiers .npz produits par data_pipeline.py.

Flux attendu AVANT d'utiliser ce module :
  python -m training.data_pipeline --kaggle data/sudoku.csv
  # ou
  python -m training.data_pipeline --generate

Les fichiers .npz contiennent :
  puzzles    (N, 81) int8  — 0=vide, 1-9=indice
  solutions  (N, 81) int8  — 1-9
  clues      (N,)    int8  — nombre d'indices
  difficulty (N,)    int8  — 0=easy…3=expert
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from training.data_pipeline import run_pipeline


class SudokuDataset(Dataset):
    """
    Dataset PyTorch chargé depuis un fichier .npz pré-traité.
    Si le fichier n'existe pas, lance automatiquement le pipeline.
    """

    def __init__(self, npz_path: str, max_size: int = None,
                 difficulty_filter: list[int] = None):
        """
        Args:
            npz_path:           Chemin vers train.npz / val.npz / test.npz
            max_size:           Limite optionnelle sur le nombre d'exemples
            difficulty_filter:  Ex. [2, 3] pour hard + expert uniquement
                                (0=easy, 1=medium, 2=hard, 3=expert)
        """
        if not Path(npz_path).exists():
            raise FileNotFoundError(
                f"{npz_path} non trouvé.\n"
                "Lancez d'abord le pipeline :\n"
                "  python -m training.data_pipeline --kaggle data/sudoku.csv\n"
                "  # ou : python -m training.data_pipeline --generate"
            )

        data = np.load(npz_path)
        puzzles    = data["puzzles"].astype(np.int64)
        solutions  = data["solutions"].astype(np.int64)
        difficulty = data.get("difficulty", np.zeros(len(puzzles), dtype=np.int64))

        # Filtre par difficulté
        if difficulty_filter is not None:
            mask = np.isin(difficulty, difficulty_filter)
            puzzles, solutions, difficulty = (
                puzzles[mask], solutions[mask], difficulty[mask]
            )

        # Limite
        if max_size:
            puzzles    = puzzles[:max_size]
            solutions  = solutions[:max_size]
            difficulty = difficulty[:max_size]

        self.puzzles    = puzzles
        self.solutions  = solutions
        self.difficulty = difficulty
        print(f"Chargé : {npz_path} → {len(self.puzzles):,} puzzles")

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.puzzles[idx],   dtype=torch.long),
            torch.tensor(self.solutions[idx], dtype=torch.long),
        )

    def stats(self) -> dict:
        clues = (self.puzzles != 0).sum(axis=1)
        diff_names = ["easy", "medium", "hard", "expert"]
        return {
            "size":      len(self.puzzles),
            "clues_mean": float(clues.mean()),
            "clues_std":  float(clues.std()),
            "clues_min":  int(clues.min()),
            "clues_max":  int(clues.max()),
            "difficulty": {
                diff_names[d]: int((self.difficulty == d).sum())
                for d in range(4)
            },
        }


def _ensure_data(out_dir: str, kaggle_csv: str = None) -> str:
    """
    Vérifie si les fichiers .npz existent, sinon lance le pipeline.
    Retourne le préfixe des fichiers ('kaggle' ou 'generated').
    """
    out = Path(out_dir)
    prefix = "kaggle" if (kaggle_csv and Path(kaggle_csv).exists()) else "generated"

    train_path = out / f"{prefix}_train.npz"
    if not train_path.exists():
        print(f"[INFO] {train_path} non trouvé — lancement du pipeline automatique...")
        run_pipeline(kaggle_csv=kaggle_csv, generate=(kaggle_csv is None), out_dir=out_dir)

    return prefix


def build_loaders(
    batch_size:   int  = 512,
    num_workers:  int  = 0,
    kaggle_csv:   str  = None,
    data_dir:     str  = "data",
    max_train:    int  = None,
    max_val:      int  = None,
    max_test:     int  = None,
    difficulty_filter: list[int] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construit les DataLoaders en chargeant les .npz pré-traités.

    Si les fichiers n'existent pas, lance data_pipeline automatiquement.

    Args:
        kaggle_csv:        data/sudoku.csv — si fourni, priorité sur la génération
        difficulty_filter: [0,1,2,3] → filtrer par niveau (None = tous)
    """
    prefix = _ensure_data(data_dir, kaggle_csv)
    out    = Path(data_dir)

    def load(split: str, max_size: int) -> SudokuDataset:
        path = str(out / f"{prefix}_{split}.npz")
        return SudokuDataset(path, max_size=max_size,
                             difficulty_filter=difficulty_filter)

    train_ds = load("train", max_train)
    val_ds   = load("val",   max_val)
    test_ds  = load("test",  max_test)

    # Afficher les stats du train
    s = train_ds.stats()
    print(f"\nTrain stats :")
    print(f"  {s['size']:,} puzzles | "
          f"indices {s['clues_mean']:.1f}±{s['clues_std']:.1f} "
          f"[{s['clues_min']}-{s['clues_max']}]")
    print(f"  Difficulté : {s['difficulty']}")

    pin = torch.cuda.is_available()

    def make(ds: SudokuDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=(num_workers > 0),
        )

    return make(train_ds, True), make(val_ds, False), make(test_ds, False)
