"""
Point d'entrée unique — lance TOUT le projet en une seule commande.

  python run.py                        # pipeline complet + UI
  python run.py --quick                # test rapide (~3 min)
  python run.py --kaggle data/sudoku.csv  # avec dataset Kaggle
  python run.py --skip-train           # juste l'API + UI (si déjà entraîné)
  python run.py --api-only             # API uniquement, pas de UI
"""

import sys
import os
import time
import signal
import argparse
import subprocess
from pathlib import Path

# On Windows, npm/npx are .cmd wrappers — use shell=True or the .cmd name.
NPM = "npm.cmd" if sys.platform == "win32" else "npm"

ROOT    = Path(__file__).parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       SUDOKU AI — Démarrage complet automatique          ║
╚══════════════════════════════════════════════════════════╝
"""


def run(cmd: list, cwd=None, check=True, silent=False) -> subprocess.CompletedProcess:
    kwargs = dict(cwd=cwd, check=check)
    if silent:
        kwargs.update(capture_output=True)
    return subprocess.run(cmd, **kwargs)


def check_tools():
    errors = []
    try:
        v = subprocess.check_output([sys.executable, "--version"], text=True).strip()
        print(f"  ✓ {v}")
    except Exception:
        errors.append("Python introuvable")

    try:
        v = subprocess.check_output(["node", "--version"], text=True).strip()
        print(f"  ✓ Node.js {v}")
    except Exception:
        errors.append("Node.js introuvable — installez depuis nodejs.org")

    try:
        v = subprocess.check_output([NPM, "--version"], text=True).strip()
        print(f"  ✓ npm {v}")
    except Exception:
        errors.append("npm introuvable — Node.js installé mais npm absent du PATH")

    if errors:
        print("\n  ERREURS :")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def install_frontend():
    print("\n  Installation des packages Node.js...")
    if not (FRONTEND / "node_modules").exists():
        run([NPM, "install", "--silent"], cwd=FRONTEND)
        print("  ✓ node_modules installé")
    else:
        print("  ✓ node_modules déjà présent")


def start_backend(args) -> subprocess.Popen:
    """Lance le backend (entraînement + API) dans un subprocess."""
    cmd = [sys.executable, "auto.py", "--serve"]
    if args.quick:        cmd.append("--quick")
    if args.skip_train:   cmd.append("--skip-train")
    if args.skip_existing:cmd.append("--skip-existing")
    if args.kaggle:       cmd += ["--kaggle", args.kaggle]
    if args.model:        cmd += ["--model", args.model]
    if args.epochs:       cmd += ["--epochs", str(args.epochs)]

    print(f"  $ {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=BACKEND)


def start_frontend() -> subprocess.Popen:
    """Lance le frontend Next.js."""
    print(f"  $ {NPM} run dev")
    return subprocess.Popen([NPM, "run", "dev"], cwd=FRONTEND)


def wait_for_api(timeout: int = 120) -> bool:
    """Attend que l'API soit disponible sur le port 8000."""
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8000/", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    print(BANNER)

    pa = argparse.ArgumentParser(description="Sudoku AI — lancement complet automatique")
    pa.add_argument("--kaggle",        type=str, default=None,
                    help="Chemin vers data/sudoku.csv")
    pa.add_argument("--quick",         action="store_true",
                    help="Mode rapide : 10k puzzles, 5 époques")
    pa.add_argument("--skip-train",    action="store_true",
                    help="Saute l'entraînement, lance juste API + UI")
    pa.add_argument("--skip-existing", action="store_true",
                    help="Ne ré-entraîne pas les modèles déjà entraînés")
    pa.add_argument("--model",         type=str, default=None,
                    help="Entraîner un seul modèle")
    pa.add_argument("--epochs",        type=int, default=None)
    pa.add_argument("--api-only",      action="store_true",
                    help="Démarre l'API uniquement (pas de frontend)")
    pa.add_argument("--aug",           type=int, default=3)
    args = pa.parse_args()

    procs = []

    def cleanup(sig=None, frame=None):
        print("\n  Arrêt des processus...")
        for p in procs:
            try: p.terminate()
            except Exception: pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ── 1. Outils ──
    print("[1] Vérification des outils...")
    check_tools()

    # ── 2. Frontend deps ──
    if not args.api_only:
        print("\n[2] Frontend...")
        install_frontend()

    # ── 3. Backend (entraînement + API) ──
    print("\n[3] Backend (entraînement + API)...")
    backend_proc = start_backend(args)
    procs.append(backend_proc)

    # ── 4. Frontend ──
    if not args.api_only:
        print("\n[4] Attente de l'API...")
        if wait_for_api(timeout=300):
            print("  ✓ API disponible sur http://localhost:8000")
            print("\n[5] Frontend Next.js...")
            frontend_proc = start_frontend()
            procs.append(frontend_proc)
            time.sleep(5)
            print("""
  ╔════════════════════════════════════════════════════╗
  ║  ✅ Sudoku AI est prêt !                           ║
  ║                                                    ║
  ║  Interface  → http://localhost:3001                ║
  ║  API        → http://localhost:8000                ║
  ║  API Docs   → http://localhost:8000/docs           ║
  ║                                                    ║
  ║  Ctrl+C pour arrêter                               ║
  ╚════════════════════════════════════════════════════╝
            """)
        else:
            print("  ✗ L'API n'a pas démarré dans les délais.")
            cleanup()

    # Attendre la fin
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
