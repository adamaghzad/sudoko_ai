"""
Pipeline entièrement automatique — une seule commande fait tout.

Étapes exécutées dans l'ordre :
  1. Vérification de l'environnement (Python, PyTorch, GPU)
  2. Installation des dépendances manquantes
  3. Détection du dataset (Kaggle CSV ou génération automatique)
  4. Nettoyage + split + augmentation (data_pipeline)
  5. Entraînement des 6 modèles (MLP, CNN, RNN, LSTM, GRU, Hybrid)
  6. Évaluation finale et tableau comparatif
  7. (Optionnel) Démarrage de l'API FastAPI

Usage :
  python auto.py                         # tout automatique
  python auto.py --kaggle data/sudoku.csv  # avec dataset Kaggle
  python auto.py --quick                 # mode test (2 min)
  python auto.py --serve                 # entraîne puis lance l'API
  python auto.py --skip-train            # saute l'entraînement, lance juste l'API
"""

import sys
import os
import json
import time
import subprocess
import argparse
from pathlib import Path

# S'assurer qu'on est dans le bon dossier
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║           SUDOKU AI — Pipeline Automatique               ║
║      MLP · CNN · RNN · LSTM · GRU · Hybrid              ║
║                EMSI Deep Learning 2025-2026              ║
╚══════════════════════════════════════════════════════════╝
"""


# ── Étape 1 : Environnement ───────────────────────────────────────────────────

def check_environment() -> dict:
    print("\n[1/6] Vérification de l'environnement...")
    info = {}

    # Python
    info["python"] = sys.version
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        _fail(f"Python 3.10+ requis. Version actuelle : {major}.{minor}")
    print(f"  ✓ Python {major}.{minor}")

    # PyTorch
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"]  = torch.cuda.is_available()
        gpu_name      = torch.cuda.get_device_name(0) if info["cuda"] else "N/A"
        info["gpu"]   = gpu_name
        print(f"  ✓ PyTorch {torch.__version__}")
        if info["cuda"]:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU : {gpu_name}  ({vram:.1f} GB VRAM)")
        else:
            print(f"  ⚠ GPU non détecté — entraînement sur CPU (plus lent)")
    except ImportError:
        info["torch"] = None
        info["cuda"]  = False
        print("  ✗ PyTorch non installé → sera installé à l'étape 2")

    return info


# ── Étape 2 : Dépendances ─────────────────────────────────────────────────────

def install_dependencies():
    print("\n[2/6] Installation des dépendances...")
    req = Path("requirements.txt")
    if not req.exists():
        _fail("requirements.txt introuvable")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Avertissement pip : {result.stderr[:300]}")
    else:
        print("  ✓ Toutes les dépendances installées")


# ── Étape 3 : Dataset ─────────────────────────────────────────────────────────

def _ensure_package(import_name: str, pip_name: str = None) -> bool:
    """Installe un package pip si absent. Retourne True si disponible."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        pip_pkg = pip_name or import_name
        print(f"  Installation de {pip_pkg}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_pkg, "-q"],
                check=True, capture_output=True, timeout=120,
            )
            return True
        except Exception as e:
            print(f"  ✗ Impossible d'installer {pip_pkg} : {e}")
            return False


def _try_kaggle_download(dest: Path) -> bool:
    """Télécharge via Kaggle API (kaggle.json ou variables KAGGLE_USERNAME/KAGGLE_KEY)."""
    import shutil

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_file = kaggle_json.exists()
    has_env  = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if not has_file and not has_env:
        print("  ℹ Kaggle : aucune credentials détectées")
        print("    Pour utiliser le dataset 1 million de puzzles (recommandé) :")
        print("    1. Créez un compte sur kaggle.com")
        print("    2. Profil → Settings → API → Create New Token → kaggle.json")
        print("    3. Placez-le dans ~/.kaggle/kaggle.json")
        print("    4. Ou définissez KAGGLE_USERNAME + KAGGLE_KEY (variables d'env)")
        return False

    # Crée kaggle.json depuis les variables d'env si le fichier est absent
    if not has_file and has_env:
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        kaggle_json.write_text(_json.dumps({
            "username": os.environ["KAGGLE_USERNAME"],
            "key":      os.environ["KAGGLE_KEY"],
        }))
        try:
            os.chmod(kaggle_json, 0o600)
        except Exception:
            pass
        print("  ✓ kaggle.json créé depuis KAGGLE_USERNAME / KAGGLE_KEY")

    if not _ensure_package("kaggle"):
        return False

    print("  Téléchargement Kaggle (bryanpark/sudoku ~113 MB)...")

    # Essaie le CLI kaggle
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "bryanpark/sudoku",
             "-p", str(dest.parent),
             "--unzip", "-q"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            for cand in [dest, dest.parent / "sudoku.csv", dest.parent / "Sudoku.csv"]:
                if cand.exists():
                    if cand != dest:
                        shutil.move(str(cand), str(dest))
                    size_mb = dest.stat().st_size / 1e6
                    print(f"  ✓ Dataset Kaggle téléchargé ({size_mb:.0f} MB)")
                    return True
        else:
            stderr = (result.stderr or result.stdout or "")[:300]
            print(f"  ✗ kaggle CLI : {stderr}")
    except subprocess.TimeoutExpired:
        print("  ✗ Timeout kaggle CLI (>10 min)")
    except FileNotFoundError:
        pass  # CLI absent du PATH, on essaie l'API Python

    # Essaie l'API Python kaggle
    try:
        import kaggle as _kgl  # type: ignore[import]
        _kgl.api.authenticate()
        print("  Tentative via API Python kaggle...")
        _kgl.api.dataset_download_files(
            "bryanpark/sudoku",
            path=str(dest.parent),
            unzip=True,
            quiet=False,
        )
        for cand in [dest, dest.parent / "sudoku.csv", dest.parent / "Sudoku.csv"]:
            if cand.exists():
                if cand != dest:
                    shutil.move(str(cand), str(dest))
                size_mb = dest.stat().st_size / 1e6
                print(f"  ✓ Dataset Kaggle téléchargé ({size_mb:.0f} MB)")
                return True
    except Exception as e:
        print(f"  ✗ API Python kaggle : {e}")

    return False


def _normalize_sudoku_df(df):
    """
    Normalise un DataFrame quelconque en colonnes (quizzes, solutions) 81-char.
    Retourne None si le format est incompatible.
    """
    import pandas as pd
    cols = list(df.columns)
    q_col = next((c for c in cols if c.lower() in
                   ("puzzle", "quizzes", "quiz", "board", "input", "question")), None)
    s_col = next((c for c in cols if c.lower() in
                   ("solution", "solutions", "output", "answer", "solved")), None)
    if not q_col or not s_col:
        return None
    df = df[[q_col, s_col]].rename(columns={q_col: "quizzes", s_col: "solutions"})
    df = df.astype(str)
    mask = (df["quizzes"].str.len() == 81) & (df["solutions"].str.len() == 81)
    df = df[mask].reset_index(drop=True)
    return df if len(df) >= 1000 else None


def _try_huggingface_download(dest: Path) -> bool:
    """Télécharge un dataset Sudoku depuis HuggingFace Hub (sans auth)."""
    if not _ensure_package("requests"):
        return False
    if not _ensure_package("pandas"):
        return False

    import requests
    import pandas as pd

    # Parquet publics HuggingFace — CDN direct, sans authentification
    parquet_sources = [
        (
            "https://huggingface.co/datasets/tdolan21/sudoku_puzzles/resolve/main/"
            "data/train-00000-of-00001-*.parquet",
            "tdolan21/sudoku_puzzles",
        ),
        (
            "https://huggingface.co/datasets/Falah/sudoku/resolve/main/"
            "data/train-00000-of-00001.parquet",
            "Falah/sudoku",
        ),
    ]

    for url_pattern, name in parquet_sources:
        # Try the URL without glob first; fall back to numbered variant
        for url in [url_pattern.replace("*", ""), url_pattern.replace("-*", "-a60dfeed9f40c44f")]:
            try:
                print(f"  HuggingFace : {name}...")
                resp = requests.get(url, timeout=30, stream=True)
                if resp.status_code != 200:
                    continue

                tmp = dest.with_suffix(".parquet.tmp")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)

                df = pd.read_parquet(tmp)
                tmp.unlink(missing_ok=True)
                df = _normalize_sudoku_df(df)
                if df is None:
                    continue

                df.to_csv(str(dest), index=False)
                size_mb = dest.stat().st_size / 1e6
                print(f"  ✓ {name} ({len(df):,} puzzles, {size_mb:.1f} MB)")
                return True

            except Exception:
                if dest.with_suffix(".parquet.tmp").exists():
                    dest.with_suffix(".parquet.tmp").unlink(missing_ok=True)

    return False


def download_dataset(kaggle_csv: str = None, data_dir: str = "data") -> str | None:
    """
    Localise ou télécharge automatiquement le dataset Sudoku.
    Ordre de priorité :
      1. Fichier déjà présent (--kaggle ou data/sudoku.csv)
      2. Téléchargement Kaggle (si kaggle.json ou KAGGLE_* env vars)
      3. Téléchargement HuggingFace (sans auth)
      4. Génération par backtracking (fallback ultime)
    """
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "sudoku.csv"

    # 1. Fichier déjà présent
    for path in [kaggle_csv, str(dest), str(out_dir / "Sudoku.csv")]:
        if path and Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            print(f"  ✓ Dataset déjà présent : {path}  ({size_mb:.1f} MB)")
            return path

    print("  Dataset non trouvé — téléchargement automatique...")

    # 2. Kaggle
    if _try_kaggle_download(dest):
        return str(dest)

    # 3. HuggingFace
    if _try_huggingface_download(dest):
        return str(dest)

    # 4. Génération par backtracking
    print("  ℹ Tous les téléchargements ont échoué → génération par backtracking")
    print("    (Pour de meilleures performances, placez data/sudoku.csv depuis Kaggle)")
    return None


def check_data_ready(csv_path: str = None, data_dir: str = "data") -> bool:
    """Vérifie si les .npz existent déjà."""
    prefix = "kaggle" if csv_path else "generated"
    out = Path(data_dir)
    return all((out / f"{prefix}_{s}.npz").exists()
               for s in ["train", "val", "test"])


def prepare_dataset(csv_path: str = None, quick: bool = False,
                    data_dir: str = "data", aug: int = 3):
    print("  Préparation du dataset...")

    if check_data_ready(csv_path, data_dir):
        print("  ✓ Fichiers .npz déjà présents — pipeline de données ignoré")
        return

    from training.data_pipeline import run_pipeline

    generating = csv_path is None
    if quick:
        train_size, aug_factor = 10_000, 1
    elif generating:
        # Backtracking is slow: limit to ~28k puzzles generated (~4 min on CPU)
        train_size, aug_factor = 20_000, 2
        print("  ℹ Mode génération — dataset réduit (20k train × 2 aug).")
        print("    Pour un dataset complet, placez data/sudoku.csv depuis Kaggle.")
    else:
        train_size, aug_factor = 100_000, aug

    run_pipeline(
        kaggle_csv=csv_path,
        generate=generating,
        train_size=train_size,
        max_rows=20_000 if quick else None,
        aug_factor=aug_factor,
        out_dir=data_dir,
        force=False,
    )
    print("  ✓ Dataset prêt")


# ── Étape 4-5 : Entraînement ──────────────────────────────────────────────────

def train_models(csv_path: str = None, quick: bool = False,
                 skip_existing: bool = False, epochs: int = None,
                 model_filter: str = None, data_dir: str = "data"):
    print("\n[4/6] Entraînement des modèles...")

    import random, numpy as np
    import torch

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, WEIGHTS_DIR
    from training.dataset  import build_loaders
    from training.trainer  import train_model, evaluate_model
    from models.mlp_solver    import MLPSolver
    from models.cnn_solver    import CNNSolver
    from models.rnn_solver    import RNNSolver
    from models.lstm_solver   import LSTMSolver
    from models.gru_solver    import GRUSolver
    from models.hybrid_solver import HybridSolver

    n_epochs = (5 if quick else (epochs or EPOCHS))

    train_loader, val_loader, test_loader = build_loaders(
        batch_size=BATCH_SIZE,
        kaggle_csv=csv_path,
        data_dir=data_dir,
        max_train=10_000 if quick else None,
        max_val=1_000    if quick else None,
        max_test=500     if quick else None,
    )

    all_models = [
        MLPSolver(hidden_sizes=[1024, 512, 256, 128], dropout=0.3),
        CNNSolver(base_channels=64, num_blocks=6, dropout=0.2),
        RNNSolver(embed_dim=16, hidden_size=256, num_layers=2, dropout=0.3),
        LSTMSolver(embed_dim=32, hidden_size=256, num_layers=2, dropout=0.3),
        GRUSolver(embed_dim=32, hidden_size=256, num_layers=2, dropout=0.3),
        HybridSolver(cnn_channels=64, hidden_size=256, num_lstm_layers=2, dropout=0.3),
    ]

    if model_filter:
        all_models = [m for m in all_models if m.name.lower() == model_filter.lower()]

    weights_dir = Path(WEIGHTS_DIR)
    results = {}

    for i, model in enumerate(all_models):
        name = model.name.lower()
        wp   = weights_dir / f"{name}.pt"

        if skip_existing and wp.exists():
            print(f"\n  [{i+1}/{len(all_models)}] {model.name} — IGNORÉ (poids existants)")
            continue

        print(f"\n  [{i+1}/{len(all_models)}] {model.name}")
        t0 = time.time()

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=n_epochs,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
        )
        test_m = evaluate_model(model, test_loader)
        elapsed = time.time() - t0

        results[name] = {"history": history, "test_metrics": test_m, "time_s": elapsed}
        print(f"  ✓ {model.name} — PuzzleAcc={test_m['puzzle_acc']:.3f} "
              f"CellAcc={test_m['cell_acc']:.3f}  ({elapsed/60:.1f} min)")

    if results:
        p = weights_dir / "training_summary.json"
        with open(p, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ── Étape 6 : Évaluation ──────────────────────────────────────────────────────

def print_final_report(results: dict):
    if not results:
        print("\n[6/6] Aucun modèle entraîné dans cette session.")
        return

    print(f"\n[6/6] Résultats finaux")
    print(f"{'='*65}")
    print(f"  {'Modèle':<10} {'CellAcc':>10} {'PuzzleAcc':>12} {'Temps':>10}")
    print(f"  {'-'*57}")
    best = max(results.items(), key=lambda x: x[1]["test_metrics"]["puzzle_acc"])
    for name, res in results.items():
        m   = res["test_metrics"]
        t   = res.get("time_s", 0)
        tag = " ← meilleur" if name == best[0] else ""
        print(f"  {name.upper():<10} {m['cell_acc']:>10.4f} {m['puzzle_acc']:>12.4f} "
              f"{t/60:>8.1f}m{tag}")
    print(f"{'='*65}")


# ── Étape 7 : Démarrage API ───────────────────────────────────────────────────

def start_api():
    print("\n[7/7] Démarrage de l'API FastAPI...")
    print("  → http://localhost:8000")
    print("  → http://localhost:8000/docs  (documentation interactive)")
    print("  Ctrl+C pour arrêter\n")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Pipeline Sudoku AI entièrement automatique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa.add_argument("--kaggle",       type=str, default=None, metavar="CSV",
                    help="Chemin vers sudoku.csv (Kaggle). Auto-détecté si absent.")
    pa.add_argument("--quick",        action="store_true",
                    help="Mode rapide : 10k puzzles, 5 époques (~2 min)")
    pa.add_argument("--serve",        action="store_true",
                    help="Lance l'API après l'entraînement")
    pa.add_argument("--skip-train",   action="store_true",
                    help="Saute l'entraînement (lance juste l'API)")
    pa.add_argument("--skip-existing",action="store_true",
                    help="Ne ré-entraîne pas les modèles déjà entraînés")
    pa.add_argument("--model",        type=str, default=None,
                    help="Entraîner un seul modèle : mlp|cnn|rnn|lstm|gru|hybrid")
    pa.add_argument("--epochs",       type=int, default=None)
    pa.add_argument("--aug",          type=int, default=3,
                    help="Facteur d'augmentation des données (défaut: 3)")
    pa.add_argument("--data-dir",     type=str, default="data")
    args = pa.parse_args()

    print(BANNER)

    # 1. Environnement
    check_environment()

    # 2. Dépendances
    install_dependencies()

    # Ré-import après installation
    import torch  # noqa: F401

    if args.skip_train:
        print("\n[INFO] Entraînement ignoré (--skip-train)")
        if args.serve:
            start_api()
        return

    if args.serve:
        # ── Mode API : démarrage immédiat, entraînement en arrière-plan ──────
        print("\n[INFO] L'API démarre immédiatement.")
        print("       Dataset + entraînement lancés en arrière-plan.\n")

        import threading

        def _background():
            try:
                print("\n[BG] Dataset Sudoku...")
                csv_path = download_dataset(args.kaggle, args.data_dir)
                prepare_dataset(csv_path, args.quick, args.data_dir, args.aug)

                print("\n[BG] Entraînement des modèles...")
                results = train_models(
                    csv_path=csv_path,
                    quick=args.quick,
                    skip_existing=args.skip_existing,
                    epochs=args.epochs,
                    model_filter=args.model,
                    data_dir=args.data_dir,
                )
                print_final_report(results)

                # Demande à l'API de recharger les poids
                import urllib.request
                try:
                    urllib.request.urlopen(
                        "http://localhost:8000/_reload_models", timeout=5
                    )
                    print("\n  ✓ Modèles rechargés dans l'API")
                except Exception:
                    print("\n  ℹ Entraînement terminé — rechargez l'API pour activer les modèles")
            except Exception as exc:
                print(f"\n  ✗ Erreur arrière-plan : {exc}")

        t = threading.Thread(target=_background, daemon=True)
        t.start()
        start_api()   # bloque jusqu'à Ctrl+C

    else:
        # ── Mode standalone : pipeline séquentiel ────────────────────────────
        t_start = time.time()
        print("\n[3/6] Dataset Sudoku...")
        csv_path = download_dataset(args.kaggle, args.data_dir)
        prepare_dataset(csv_path, args.quick, args.data_dir, args.aug)

        results = train_models(
            csv_path=csv_path,
            quick=args.quick,
            skip_existing=args.skip_existing,
            epochs=args.epochs,
            model_filter=args.model,
            data_dir=args.data_dir,
        )
        print_final_report(results)
        print(f"\n  Temps total : {(time.time()-t_start)/60:.1f} min")
        print("""
  ┌─────────────────────────────────────────────────┐
  │  Pour lancer l'API :  python auto.py --serve    │
  │  Pour lancer le UI :  cd ../frontend             │
  │                        npm run dev               │
  │  Ou tout d'un coup :  cd .. && python run.py    │
  └─────────────────────────────────────────────────┘
        """)


def _fail(msg: str):
    print(f"\n  ✗ ERREUR : {msg}")
    sys.exit(1)


if __name__ == "__main__":
    main()
