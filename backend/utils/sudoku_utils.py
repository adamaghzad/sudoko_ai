"""
Utilitaires Sudoku : génération, validation, augmentation, classification de difficulté.

Problèmes corrigés vs version initiale :
  1. La génération n'assurait pas l'unicité de la solution — corrigé.
  2. Pas de classification de difficulté — ajoutée.
  3. Pas d'augmentation de données — ajoutée (8 symétries + permutations).
"""
import random
import numpy as np
from typing import Optional
from enum import Enum


# ── Difficulté ────────────────────────────────────────────────────────────────

class Difficulty(Enum):
    EASY   = "easy"    # 36-45 indices
    MEDIUM = "medium"  # 28-35 indices
    HARD   = "hard"    # 20-27 indices
    EXPERT = "expert"  # 17-19 indices  (minimum théorique prouvé = 17)

DIFFICULTY_RANGES = {
    Difficulty.EASY:   (36, 45),
    Difficulty.MEDIUM: (28, 35),
    Difficulty.HARD:   (20, 27),
    Difficulty.EXPERT: (17, 19),
}

def difficulty_from_clues(n: int) -> Difficulty:
    if n >= 36: return Difficulty.EASY
    if n >= 28: return Difficulty.MEDIUM
    if n >= 20: return Difficulty.HARD
    return Difficulty.EXPERT


# ── Validation ────────────────────────────────────────────────────────────────

def validate_puzzle_input(puzzle: list[int]) -> tuple[bool, str]:
    """
    Checks the input puzzle for contradictions (duplicate digits in a row/col/box).
    Returns (True, "") if valid, or (False, error_message) if contradictions found.
    Does NOT check for solution existence — only checks given clues are consistent.
    """
    board = np.array(puzzle, dtype=np.int64).reshape(9, 9)
    for i in range(9):
        row = [int(v) for v in board[i] if v != 0]
        if len(row) != len(set(row)):
            return False, f"Duplication dans la ligne {i + 1}"
        col = [int(v) for v in board[:, i] if v != 0]
        if len(col) != len(set(col)):
            return False, f"Duplication dans la colonne {i + 1}"
    for br in range(3):
        for bc in range(3):
            box = [int(v) for v in board[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten() if v != 0]
            if len(box) != len(set(box)):
                return False, f"Duplication dans le bloc ({br+1},{bc+1})"
    return True, ""


def is_valid_placement(board: list[list[int]], row: int, col: int, num: int) -> bool:
    """Vérifie qu'un chiffre peut être placé en (row, col) sans violer les règles."""
    if num in board[row]:
        return False
    if num in (board[r][col] for r in range(9)):
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    return not any(
        board[r][c] == num
        for r in range(br, br + 3)
        for c in range(bc, bc + 3)
    )

# Alias utilisé dans le projet
is_valid = is_valid_placement


def count_solutions(board: list[list[int]], limit: int = 2) -> int:
    """
    Compte le nombre de solutions (s'arrête à `limit` pour la performance).
    Utilisé pour garantir l'unicité : retourne 1 si unique, 2+ sinon.
    """
    count = [0]

    def _solve():
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    for num in range(1, 10):
                        if is_valid_placement(board, r, c, num):
                            board[r][c] = num
                            _solve()
                            board[r][c] = 0
                            if count[0] >= limit:
                                return
                    return
        count[0] += 1

    _solve()
    return count[0]


def has_unique_solution(board: list[list[int]]) -> bool:
    """Retourne True si le puzzle possède exactement une solution."""
    b = [row[:] for row in board]
    return count_solutions(b, limit=2) == 1


def is_valid_solution(puzzle: np.ndarray, solution: np.ndarray) -> bool:
    """Vérifie qu'une solution est complète, valide, et cohérente avec le puzzle."""
    sol = solution.reshape(9, 9)
    puz = puzzle.reshape(9, 9)
    # Clues preserved
    for r in range(9):
        for c in range(9):
            if puz[r][c] != 0 and puz[r][c] != sol[r][c]:
                return False
    # Rows, cols, boxes
    for i in range(9):
        if set(sol[i]) != set(range(1, 10)):
            return False
        if set(sol[:, i]) != set(range(1, 10)):
            return False
    for br in range(3):
        for bc in range(3):
            if set(sol[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()) != set(range(1, 10)):
                return False
    return True


# ── Génération ────────────────────────────────────────────────────────────────

def solve_backtrack(board: list[list[int]], randomize: bool = True) -> bool:
    """Backtracking avec ordre aléatoire des chiffres (génération) ou déterministe (validation)."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                nums = list(range(1, 10))
                if randomize:
                    random.shuffle(nums)
                for num in nums:
                    if is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if solve_backtrack(board, randomize):
                            return True
                        board[row][col] = 0
                return False
    return True

# Alias non-aléatoire pour la validation
def solve_deterministic(board: list[list[int]]) -> bool:
    return solve_backtrack(board, randomize=False)


def generate_solved_board() -> list[list[int]]:
    """Génère une grille Sudoku complète et valide."""
    board = [[0] * 9 for _ in range(9)]
    solve_backtrack(board, randomize=True)
    return board


def create_puzzle(solved: list[list[int]], num_clues: int = 30,
                  ensure_unique: bool = True) -> Optional[list[list[int]]]:
    """
    Crée un puzzle en retirant des cellules de la grille complète.

    Si ensure_unique=True, garantit qu'une seule solution existe.
    Retourne None si impossible d'atteindre num_clues avec unicité.
    """
    puzzle = [row[:] for row in solved]
    cells = list(range(81))
    random.shuffle(cells)

    removed = 0
    target = 81 - num_clues

    for idx in cells:
        if removed >= target:
            break
        r, c = idx // 9, idx % 9
        saved = puzzle[r][c]
        puzzle[r][c] = 0

        if ensure_unique and not has_unique_solution(puzzle):
            puzzle[r][c] = saved  # restaurer si non unique
        else:
            removed += 1

    actual_clues = sum(1 for row in puzzle for v in row if v != 0)
    if actual_clues > num_clues + 3:
        return None  # trop difficile à atteindre
    return puzzle


# ── Augmentation de données ───────────────────────────────────────────────────
#
# Un puzzle Sudoku possède 8 symétries (groupe diédral D4) qui préservent
# la validité de la structure :
#   - Transpositions (2)          → swap lignes↔colonnes
#   - Rotations 90/180/270° (3)
#   - Permutations de bandes      → permuter les 3 bandes de lignes/colonnes (3!×3! = 36)
#   - Permutations de chiffres    → remapper 1-9 (9! = 362880)
# En pratique on utilise les transformations géométriques + permutation de chiffres.

def _apply_transform(board: np.ndarray, transform_id: int) -> np.ndarray:
    """Applique l'une des 8 transformations géométriques."""
    b = board.reshape(9, 9)
    ops = [
        lambda x: x,                          # identité
        lambda x: np.rot90(x, 1),             # 90°
        lambda x: np.rot90(x, 2),             # 180°
        lambda x: np.rot90(x, 3),             # 270°
        lambda x: np.flipud(x),               # flip vertical
        lambda x: np.fliplr(x),               # flip horizontal
        lambda x: x.T,                        # transpose
        lambda x: np.fliplr(x.T),             # anti-transpose
    ]
    return ops[transform_id % 8](b).flatten()


def _permute_digits(board: np.ndarray, mapping: list[int]) -> np.ndarray:
    """
    Remapping des chiffres 1-9 (exemple : 1→3, 2→7, …).
    mapping[i] est le nouveau chiffre pour l'ancien chiffre i+1.
    """
    out = board.copy()
    for i, m in enumerate(mapping):
        out[board == i + 1] = m
    return out


def _swap_bands(board: np.ndarray, axis: int, perm: list[int]) -> np.ndarray:
    """Permute les 3 bandes de lignes (axis=0) ou colonnes (axis=1)."""
    b = board.reshape(9, 9)
    if axis == 0:
        rows = [b[p*3:(p+1)*3] for p in perm]
        return np.vstack(rows).flatten()
    else:
        cols = [b[:, p*3:(p+1)*3] for p in perm]
        return np.hstack(cols).flatten()


def augment(puzzle: np.ndarray, solution: np.ndarray,
            n: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Génère n variantes augmentées du même puzzle.
    Chaque variante applique : transformation géométrique + permutation de bandes + remap chiffres.
    Les variantes sont toutes des puzzles Sudoku valides avec une solution valide.
    """
    results = []
    for _ in range(n):
        t = random.randint(0, 7)
        puz = _apply_transform(puzzle, t)
        sol = _apply_transform(solution, t)

        # Permuter bandes de lignes et colonnes
        bp = random.sample(range(3), 3)
        puz = _swap_bands(puz, 0, bp)
        sol = _swap_bands(sol, 0, bp)
        cp = random.sample(range(3), 3)
        puz = _swap_bands(puz, 1, cp)
        sol = _swap_bands(sol, 1, cp)

        # Remapper les chiffres
        mapping = list(range(1, 10))
        random.shuffle(mapping)
        puz = _permute_digits(puz, mapping)
        sol = _permute_digits(sol, mapping)

        results.append((puz, sol))
    return results


# ── Conversion ────────────────────────────────────────────────────────────────

def board_to_array(board: list[list[int]]) -> np.ndarray:
    return np.array(board, dtype=np.int64).flatten()

def array_to_board(arr: np.ndarray) -> list[list[int]]:
    return arr.reshape(9, 9).tolist()

def string_to_array(s: str) -> np.ndarray:
    """Convertit une chaîne de 81 chiffres en array (format Kaggle)."""
    return np.array([int(c) for c in s.strip()], dtype=np.int64)


def generate_sample(num_clues: Optional[int] = None,
                    ensure_unique: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un (puzzle, solution).

    ensure_unique=False par défaut pour la vitesse (l'unicité est une propriété humaine,
    les modèles DL peuvent apprendre sans elle). Mettre True pour des puzzles "propres".
    """
    if num_clues is None:
        num_clues = random.randint(25, 36)
    solved = generate_solved_board()
    puzzle = create_puzzle(solved, num_clues, ensure_unique=ensure_unique)
    if puzzle is None:
        return generate_sample(num_clues, ensure_unique)
    return board_to_array(puzzle), board_to_array(solved)


# ── Formatage ─────────────────────────────────────────────────────────────────

def format_board(arr: np.ndarray) -> str:
    board = arr.reshape(9, 9)
    lines = []
    for i, row in enumerate(board):
        if i % 3 == 0 and i > 0:
            lines.append("------+-------+------")
        row_str = ""
        for j, val in enumerate(row):
            if j % 3 == 0 and j > 0:
                row_str += "| "
            row_str += f"{val if val != 0 else '.'} "
        lines.append(row_str.strip())
    return "\n".join(lines)
