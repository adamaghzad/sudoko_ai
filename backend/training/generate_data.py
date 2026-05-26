"""
Script de pré-génération du dataset (optionnel).
Génère les fichiers .npz une seule fois avant l'entraînement.

Usage :
  python -m training.generate_data                        # 100k / 10k / 5k
  python -m training.generate_data --train 500000         # dataset plus grand
  python -m training.generate_data --kaggle data/sudoku.csv  # depuis Kaggle CSV
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import SudokuDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=100_000)
    parser.add_argument("--val",   type=int, default=10_000)
    parser.add_argument("--test",  type=int, default=5_000)
    parser.add_argument("--kaggle", type=str, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    Path(args.out).mkdir(exist_ok=True)
    prefix = "kaggle" if args.kaggle else "generated"

    for split, size in [("train", args.train), ("val", args.val), ("test", args.test)]:
        cache = f"{args.out}/{prefix}_{split}.npz"
        if Path(cache).exists():
            print(f"{split}: cache déjà existant → {cache}")
            continue
        print(f"\n[{split}] Génération de {size:,} puzzles...")
        SudokuDataset(
            size=size,
            kaggle_csv=args.kaggle,
            cache_path=cache,
            use_augmentation=(not args.no_augment and split == "train"),
        )
    print("\nGénération terminée.")

if __name__ == "__main__":
    main()
