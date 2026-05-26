"""
Pipeline de prétraitement (nettoyage) du dataset Sudoku.

Étapes :
  1. Chargement       — CSV Kaggle ou génération par backtracking
  2. Nettoyage        — format, valeurs, doublons, cohérence puzzle/solution
  3. Enrichissement   — calcul difficulté, nombre d'indices, flag vide
  4. Split stratifié  — Train / Val / Test avec distribution équilibrée
  5. Augmentation     — symétries géométriques + permutations (train uniquement)
  6. Sauvegarde cache — .npz compressé pour rechargement rapide
"""

import json
import random
from pathlib import Path

import numpy as np

from utils.sudoku_utils import (
    is_valid_solution, augment, difficulty_from_clues,
    Difficulty, generate_sample, string_to_array,
)


# ── 1. Chargement ─────────────────────────────────────────────────────────────

def load_kaggle_csv(csv_path: str, max_rows: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge le CSV Kaggle.
    Colonnes attendues : 'quizzes' (ou 'puzzle') + 'solutions' (ou 'solution').
    Chaque valeur est une chaîne de 81 caractères '0'-'9' (0=vide).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas")

    print(f"\n[CHARGEMENT] {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows, dtype=str)
    print(f"  Lignes brutes : {len(df):,}")

    # Détection automatique des noms de colonnes
    col_q = next((c for c in ["quizzes", "puzzle", "Quizzes", "Puzzle"] if c in df.columns), None)
    col_s = next((c for c in ["solutions", "solution", "Solutions", "Solution"] if c in df.columns), None)

    if col_q is None or col_s is None:
        raise ValueError(f"Colonnes non trouvées. Colonnes disponibles : {list(df.columns)}")

    puzzles   = df[col_q].str.strip().values
    solutions = df[col_s].str.strip().values
    return puzzles, solutions


def generate_raw(size: int, min_clues: int = 25, max_clues: int = 36,
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Génère des puzzles par backtracking (fallback sans CSV)."""
    print(f"\n[GÉNÉRATION] {size:,} puzzles par backtracking...")
    puzzles   = np.zeros((size, 81), dtype=np.int8)
    solutions = np.zeros((size, 81), dtype=np.int8)
    for i in range(size):
        nc = random.randint(min_clues, max_clues)
        p, s = generate_sample(nc, ensure_unique=False)
        puzzles[i]   = p.astype(np.int8)
        solutions[i] = s.astype(np.int8)
        if verbose and (i + 1) % 10_000 == 0:
            print(f"  {i+1:,}/{size:,}")
    return puzzles, solutions


# ── 2. Nettoyage ──────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Applique une suite de filtres sur les paires (puzzle, solution).

    Filtres appliqués (dans l'ordre) :
      ① Longueur         — 81 caractères exactement
      ② Caractères       — uniquement 0-9
      ③ Solution complète — aucun 0 dans la solution
      ④ Cohérence        — les indices du puzzle sont présents dans la solution
      ⑤ Doublons         — suppression des puzzles identiques
      ⑥ Solution valide  — lignes/colonnes/boîtes sans répétition
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.report: dict[str, int] = {}

    def _log(self, step: str, n_before: int, n_after: int):
        dropped = n_before - n_after
        self.report[step] = dropped
        if self.verbose and dropped > 0:
            print(f"  [{step}] supprimés : {dropped:,}  →  restants : {n_after:,}")

    def clean_from_strings(self, raw_puzzles: np.ndarray,
                           raw_solutions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Nettoyage de données provenant du CSV (chaînes de caractères).
        Retourne des arrays int8 de shape (N, 81).
        """
        n0 = len(raw_puzzles)
        print(f"\n[NETTOYAGE] {n0:,} puzzles bruts")

        # ① Longueur
        mask = (np.char.str_len(raw_puzzles) == 81) & (np.char.str_len(raw_solutions) == 81)
        raw_puzzles, raw_solutions = raw_puzzles[mask], raw_solutions[mask]
        self._log("longueur≠81", n0, len(raw_puzzles))

        # ② Caractères valides (0-9 uniquement)
        valid_chars = np.array([
            all(c.isdigit() for c in p) and all(c.isdigit() for c in s)
            for p, s in zip(raw_puzzles, raw_solutions)
        ])
        raw_puzzles, raw_solutions = raw_puzzles[valid_chars], raw_solutions[valid_chars]
        self._log("caractères invalides", len(raw_puzzles) + (~valid_chars).sum(), len(raw_puzzles))

        # Conversion en arrays numériques
        puzzles   = np.array([[int(c) for c in p] for p in raw_puzzles], dtype=np.int8)
        solutions = np.array([[int(c) for c in s] for s in raw_solutions], dtype=np.int8)

        return self.clean_arrays(puzzles, solutions, already_converted=True)

    def clean_arrays(self, puzzles: np.ndarray, solutions: np.ndarray,
                     already_converted: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Nettoyage de données déjà sous forme numérique int8 (N, 81).
        """
        n = len(puzzles)
        if not already_converted:
            print(f"\n[NETTOYAGE] {n:,} puzzles")

        # ③ Solution complète (aucun 0)
        sol_complete = (solutions != 0).all(axis=1)
        puzzles, solutions = puzzles[sol_complete], solutions[sol_complete]
        self._log("solution incomplète", n, len(puzzles)); n = len(puzzles)

        # ④ Valeurs dans la plage 0-9 / 1-9
        puz_valid = ((puzzles >= 0) & (puzzles <= 9)).all(axis=1)
        sol_valid = ((solutions >= 1) & (solutions <= 9)).all(axis=1)
        mask = puz_valid & sol_valid
        puzzles, solutions = puzzles[mask], solutions[mask]
        self._log("valeurs hors plage", n, len(puzzles)); n = len(puzzles)

        # ⑤ Cohérence : les indices du puzzle doivent correspondre à la solution
        clue_mask = (puzzles != 0)
        coherent = np.all((puzzles == solutions) | ~clue_mask, axis=1)
        puzzles, solutions = puzzles[coherent], solutions[coherent]
        self._log("incohérence puzzle/solution", n, len(puzzles)); n = len(puzzles)

        # ⑥ Doublons (sur les puzzles uniquement)
        puz_tuples = [tuple(row) for row in puzzles]
        seen, unique_idx = set(), []
        for i, t in enumerate(puz_tuples):
            if t not in seen:
                seen.add(t)
                unique_idx.append(i)
        unique_idx = np.array(unique_idx)
        puzzles, solutions = puzzles[unique_idx], solutions[unique_idx]
        self._log("doublons", n, len(puzzles)); n = len(puzzles)

        # ⑦ Validation Sudoku (lignes + colonnes + boîtes)
        # Vérification vectorisée rapide
        valid_sol = self._validate_solutions_vectorized(solutions)
        puzzles, solutions = puzzles[valid_sol], solutions[valid_sol]
        self._log("solution Sudoku invalide", n, len(puzzles))

        print(f"  ✓ Dataset propre : {len(puzzles):,} puzzles valides")
        return puzzles, solutions

    @staticmethod
    def _validate_solutions_vectorized(solutions: np.ndarray) -> np.ndarray:
        """Validation vectorisée : rows + cols + boxes."""
        N = len(solutions)
        S = solutions.reshape(N, 9, 9)
        expected = set(range(1, 10))

        valid = np.ones(N, dtype=bool)

        # Lignes
        for r in range(9):
            row_sets = [set(S[i, r]) for i in range(N)]
            valid &= np.array([s == expected for s in row_sets])

        # Colonnes
        for c in range(9):
            col_sets = [set(S[i, :, c]) for i in range(N)]
            valid &= np.array([s == expected for s in col_sets])

        # Boîtes 3×3
        for br in range(3):
            for bc in range(3):
                box_sets = [set(S[i, br*3:(br+1)*3, bc*3:(bc+1)*3].flatten())
                            for i in range(N)]
                valid &= np.array([s == expected for s in box_sets])

        return valid


# ── 3. Enrichissement ─────────────────────────────────────────────────────────

def enrich(puzzles: np.ndarray, solutions: np.ndarray) -> dict:
    """
    Calcule les métadonnées utiles pour l'analyse et le split stratifié.

    Retourne un dict avec :
      clues      : (N,) int8  — nombre d'indices par puzzle
      difficulty : (N,) int8  — 0=easy, 1=medium, 2=hard, 3=expert
    """
    clues = (puzzles != 0).sum(axis=1).astype(np.int8)
    diff_map = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1,
                Difficulty.HARD: 2, Difficulty.EXPERT: 3}
    difficulty = np.array([diff_map[difficulty_from_clues(int(n))] for n in clues], dtype=np.int8)
    return {"clues": clues, "difficulty": difficulty}


# ── 4. Split stratifié ────────────────────────────────────────────────────────

def stratified_split(puzzles: np.ndarray, solutions: np.ndarray,
                     metadata: dict,
                     train_ratio: float = 0.80,
                     val_ratio:   float = 0.10,
                     test_ratio:  float = 0.10,
                     seed: int = 42) -> dict:
    """
    Split Train/Val/Test stratifié par niveau de difficulté.
    Garantit que chaque split contient des puzzles de chaque niveau.

    Ratios : train=80%, val=10%, test=10% (par défaut)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    difficulty = metadata["difficulty"]
    n = len(puzzles)

    train_idx, val_idx, test_idx = [], [], []

    for d in range(4):  # 4 niveaux de difficulté
        idx = np.where(difficulty == d)[0]
        rng.shuffle(idx)
        n_d = len(idx)
        if n_d == 0:
            continue
        n_train = int(n_d * train_ratio)
        n_val   = int(n_d * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(  idx[n_train:n_train + n_val])
        test_idx.extend( idx[n_train + n_val:])

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    # Mélange final
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    print(f"\n[SPLIT] stratifié par difficulté")
    print(f"  Train : {len(train_idx):>8,}  ({len(train_idx)/n*100:.1f}%)")
    print(f"  Val   : {len(val_idx):>8,}  ({len(val_idx)/n*100:.1f}%)")
    print(f"  Test  : {len(test_idx):>8,}  ({len(test_idx)/n*100:.1f}%)")

    diff_names = ["easy", "medium", "hard", "expert"]
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        d_dist = {diff_names[d]: int((difficulty[idx] == d).sum()) for d in range(4)}
        print(f"  {split_name} difficulté : {d_dist}")

    return {
        "train": (puzzles[train_idx], solutions[train_idx],
                  {k: v[train_idx] for k, v in metadata.items()}),
        "val":   (puzzles[val_idx],   solutions[val_idx],
                  {k: v[val_idx]   for k, v in metadata.items()}),
        "test":  (puzzles[test_idx],  solutions[test_idx],
                  {k: v[test_idx]  for k, v in metadata.items()}),
    }


# ── 5. Augmentation ───────────────────────────────────────────────────────────

def apply_augmentation(puzzles: np.ndarray, solutions: np.ndarray,
                       factor: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Multiplie les données d'entraînement par `factor` via :
      • 8 transformations géométriques (rotations + réflexions)
      • Permutations de bandes de lignes/colonnes (3! × 3! = 36)
      • Remapping aléatoire des chiffres 1-9 (9! = 362 880)
    Toutes ces transformations préservent la validité Sudoku.
    """
    if factor <= 1:
        return puzzles, solutions

    print(f"\n[AUGMENTATION] ×{factor} (géométrie + permutations + remapping)")
    base_n = len(puzzles)

    aug_p = [puzzles]
    aug_s = [solutions]

    for puz, sol in zip(puzzles, solutions):
        for ap, as_ in augment(puz, sol, n=factor - 1):
            aug_p.append(ap.reshape(1, -1))
            aug_s.append(as_.reshape(1, -1))

    aug_p_arr = np.vstack(aug_p).astype(np.int8)
    aug_s_arr = np.vstack(aug_s).astype(np.int8)

    # Mélange
    idx = np.random.permutation(len(aug_p_arr))
    aug_p_arr = aug_p_arr[idx]
    aug_s_arr = aug_s_arr[idx]

    print(f"  {base_n:,} → {len(aug_p_arr):,} puzzles (+{len(aug_p_arr)-base_n:,})")
    return aug_p_arr, aug_s_arr


# ── 6. Sauvegarde et stats ────────────────────────────────────────────────────

def save_split(path: str, puzzles: np.ndarray, solutions: np.ndarray, metadata: dict):
    np.savez_compressed(path, puzzles=puzzles, solutions=solutions, **metadata)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Sauvegardé : {path}  ({len(puzzles):,} puzzles, {size_mb:.1f} MB)")


def compute_stats(splits: dict, cleaner: DataCleaner) -> dict:
    stats = {"cleaning": cleaner.report, "splits": {}}
    for name, (puz, sol, meta) in splits.items():
        clues = meta["clues"]
        diff  = meta["difficulty"]
        diff_names = ["easy", "medium", "hard", "expert"]
        stats["splits"][name] = {
            "size":      int(len(puz)),
            "clues_mean": float(clues.mean()),
            "clues_std":  float(clues.std()),
            "clues_min":  int(clues.min()),
            "clues_max":  int(clues.max()),
            "difficulty": {diff_names[d]: int((diff == d).sum()) for d in range(4)},
        }
    return stats


def print_stats(stats: dict):
    print("\n" + "=" * 55)
    print("  STATISTIQUES DU DATASET")
    print("=" * 55)

    print("\nNettoyage :")
    for step, dropped in stats["cleaning"].items():
        print(f"  {step:<35} -{ dropped:>6,}")

    print("\nSplits :")
    for name, s in stats["splits"].items():
        print(f"\n  [{name.upper()}]  {s['size']:,} puzzles")
        print(f"    Indices : {s['clues_mean']:.1f} ± {s['clues_std']:.1f} "
              f"[{s['clues_min']}-{s['clues_max']}]")
        total = s["size"] or 1
        for d, n in s["difficulty"].items():
            bar = "█" * int(n / total * 20)
            print(f"    {d:<8} {n:>7,}  {bar}  ({n/total*100:.1f}%)")
    print("=" * 55)
