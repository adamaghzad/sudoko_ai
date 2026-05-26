"""
Orchestrateur du pipeline complet de données.

FLUX COMPLET :
  Source (CSV ou backtracking)
    ↓
  Nettoyage (7 filtres)
    ↓
  Enrichissement (difficulté, nb indices)
    ↓
  Split stratifié (Train 80% / Val 10% / Test 10%)
    ↓
  Augmentation (train uniquement, ×3 par défaut)
    ↓
  Sauvegarde .npz + stats.json

Usage :
  python -m training.data_pipeline --kaggle data/sudoku.csv
  python -m training.data_pipeline --generate --train 100000
  python -m training.data_pipeline --kaggle data/sudoku.csv --max 500000 --aug 4
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.preprocessing import (
    load_kaggle_csv, generate_raw,
    DataCleaner, enrich, stratified_split,
    apply_augmentation, save_split, compute_stats, print_stats,
)


def run_pipeline(
    kaggle_csv:  str   = None,
    generate:    bool  = False,
    train_size:  int   = 100_000,
    max_rows:    int   = None,
    aug_factor:  int   = 3,
    out_dir:     str   = "data",
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.10,
    seed:        int   = 42,
    force:       bool  = False,
) -> dict:
    """
    Exécute le pipeline complet et retourne les statistiques.
    """
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    stats_path = out / "stats.json"
    if stats_path.exists() and not force:
        print(f"[INFO] Pipeline déjà exécuté. Stats : {stats_path}")
        print("[INFO] Utilisez --force pour relancer.")
        with open(stats_path) as f:
            return json.load(f)

    print("\n" + "=" * 55)
    print("  SUDOKU AI — PIPELINE DE DONNÉES")
    print("=" * 55)

    # ── 1. Chargement ──────────────────────────────────────────
    if kaggle_csv and Path(kaggle_csv).exists():
        raw_puz_str, raw_sol_str = load_kaggle_csv(kaggle_csv, max_rows=max_rows)
        source = "kaggle"
    else:
        if kaggle_csv:
            print(f"[AVERTISSEMENT] {kaggle_csv} non trouvé → mode génération")
        raw_puz_arr, raw_sol_arr = generate_raw(
            size=int((train_size / train_ratio) * 1.15),  # 15% de marge pour les filtres
            verbose=True,
        )
        source = "generated"

    # ── 2. Nettoyage ───────────────────────────────────────────
    cleaner = DataCleaner(verbose=True)
    if source == "kaggle":
        puzzles, solutions = cleaner.clean_from_strings(raw_puz_str, raw_sol_str)
    else:
        puzzles, solutions = cleaner.clean_arrays(raw_puz_arr, raw_sol_arr)

    # Limite optionnelle après nettoyage
    if max_rows and len(puzzles) > max_rows:
        puzzles   = puzzles[:max_rows]
        solutions = solutions[:max_rows]
        print(f"[INFO] Limité à {max_rows:,} puzzles après nettoyage")

    # ── 3. Enrichissement ──────────────────────────────────────
    print("\n[ENRICHISSEMENT] Calcul difficulté et métadonnées...")
    metadata = enrich(puzzles, solutions)

    # ── 4. Split stratifié ─────────────────────────────────────
    splits = stratified_split(
        puzzles, solutions, metadata,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # ── 5. Augmentation (train uniquement) ─────────────────────
    if aug_factor > 1:
        tr_puz, tr_sol, tr_meta = splits["train"]
        tr_puz_aug, tr_sol_aug = apply_augmentation(tr_puz, tr_sol, factor=aug_factor)
        # Recalcul des métadonnées après augmentation
        tr_meta_aug = enrich(tr_puz_aug, tr_sol_aug)
        splits["train"] = (tr_puz_aug, tr_sol_aug, tr_meta_aug)

    # ── 6. Sauvegarde ──────────────────────────────────────────
    print("\n[SAUVEGARDE]")
    prefix = source
    for split_name, (puz, sol, meta) in splits.items():
        save_split(str(out / f"{prefix}_{split_name}.npz"), puz, sol, meta)

    # Stats globales
    stats = compute_stats(splits, cleaner)
    stats["source"] = source
    stats["aug_factor"] = aug_factor

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats JSON : {stats_path}")

    print_stats(stats)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de données Sudoku AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Depuis Kaggle CSV (recommandé)
  python -m training.data_pipeline --kaggle data/sudoku.csv

  # Kaggle CSV, limité à 500k puzzles, augmentation ×4
  python -m training.data_pipeline --kaggle data/sudoku.csv --max 500000 --aug 4

  # Génération par backtracking
  python -m training.data_pipeline --generate --train 100000

  # Forcer la régénération même si le cache existe
  python -m training.data_pipeline --kaggle data/sudoku.csv --force
        """,
    )
    parser.add_argument("--kaggle",   type=str, default=None, metavar="CSV",
                        help="Chemin vers sudoku.csv (Kaggle)")
    parser.add_argument("--generate", action="store_true",
                        help="Générer par backtracking (si pas de CSV)")
    parser.add_argument("--train",    type=int,   default=100_000,
                        help="Taille cible du train set (défaut: 100 000)")
    parser.add_argument("--max",      type=int,   default=None,
                        help="Limite maximale de lignes CSV à charger")
    parser.add_argument("--aug",      type=int,   default=3,
                        help="Facteur d'augmentation train (défaut: 3, 0=désactivé)")
    parser.add_argument("--out",      type=str,   default="data",
                        help="Dossier de sortie (défaut: data/)")
    parser.add_argument("--force",    action="store_true",
                        help="Relancer même si le cache existe")
    args = parser.parse_args()

    if not args.kaggle and not args.generate:
        print("[INFO] Aucune source spécifiée. Mode génération par défaut.")
        args.generate = True

    run_pipeline(
        kaggle_csv=args.kaggle,
        generate=args.generate,
        train_size=args.train,
        max_rows=args.max,
        aug_factor=args.aug,
        out_dir=args.out,
        force=args.force,
    )


if __name__ == "__main__":
    main()
