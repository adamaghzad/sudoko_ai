"""
FastAPI Backend — Sudoku AI
Endpoints : /solve/grid | /solve/image | /models/status | /generate
"""
import base64
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import CORS_ORIGINS
from inference.solver import SudokuSolverEngine
from inference.image_processor import image_to_grid
from inference.digit_recognizer import DigitRecognizer
from utils.sudoku_utils import generate_sample, format_board


# ── État global ──────────────────────────────────────────────────────────────
engine: SudokuSolverEngine = None
recognizer: DigitRecognizer = None
_recognizer_ready = False
_recognizer_error: str = None


def _init_recognizer():
    """Runs in a background thread at startup — trains MNIST if weights absent."""
    global recognizer, _recognizer_ready, _recognizer_error
    try:
        print("Initialisation DigitCNN en arrière-plan...")
        recognizer = DigitRecognizer()
        _recognizer_ready = True
        print("DigitCNN prêt.")
    except Exception as exc:
        _recognizer_error = str(exc)
        print(f"Erreur DigitCNN: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import threading
    global engine
    print("Chargement des modèles de résolution...")
    engine = SudokuSolverEngine()
    threading.Thread(target=_init_recognizer, daemon=True).start()
    print("API prête.  (DigitCNN initialisation en arrière-plan)")
    yield
    print("Arrêt de l'API.")


def _get_recognizer() -> DigitRecognizer:
    if not _recognizer_ready:
        msg = _recognizer_error or "DigitCNN encore en cours d'initialisation, réessayez dans quelques secondes"
        raise HTTPException(503, msg)
    return recognizer


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sudoku AI API",
    description="Résolution de Sudoku par Deep Learning — MLP, CNN, RNN, LSTM, GRU, Hybrid",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class GridInput(BaseModel):
    puzzle: list[int]

    @field_validator("puzzle")
    @classmethod
    def validate_puzzle(cls, v):
        if len(v) != 81:
            raise ValueError("Le puzzle doit contenir exactement 81 valeurs")
        if any(n < 0 or n > 9 for n in v):
            raise ValueError("Chaque valeur doit être entre 0 (vide) et 9")
        return v


class TrainRequest(BaseModel):
    quick: bool = False
    model: str = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Sudoku AI API",
        "version": "1.0.0",
        "endpoints": ["/solve/grid", "/solve/image", "/models/status", "/generate"],
    }


@app.post("/solve/grid")
def solve_grid(body: GridInput):
    """Résout un puzzle fourni comme liste de 81 entiers (0=vide)."""
    if engine is None:
        raise HTTPException(503, "Moteur non initialisé")
    try:
        return engine.solve(body.puzzle)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Erreur interne: {e}")


@app.post("/solve/image")
async def solve_image(file: UploadFile = File(...)):
    """
    Pipeline complet image → puzzle → solution.
    1. Détection grille (OpenCV)
    2. Extraction cellules
    3. Reconnaissance chiffres (DigitCNN)
    4. Résolution (tous les modèles)
    """
    if engine is None:
        raise HTTPException(503, "Moteur non initialisé")

    # Raises 503 if DigitCNN not ready yet (MNIST still training in background)
    rec = _get_recognizer()

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Le fichier doit être une image")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image trop grande (max 10 MB)")

    try:
        # Étape 1 & 2 : détection + extraction
        grid_data = image_to_grid(image_bytes)
        cells = grid_data["cells"]
        empty_flags = grid_data["empty_flags"]

        # Étape 3 : reconnaissance chiffres
        cell_results = rec.predict_with_confidence(cells, empty_flags)
        puzzle = [r["digit"] for r in cell_results]

        # Étape 4 : résolution
        solve_result = engine.solve(puzzle)

        # Encode la grille détectée en base64 pour le frontend
        grid_b64 = base64.b64encode(grid_data["grid_image_bytes"]).decode()

        return {
            **solve_result,
            "ocr": {
                "cells": cell_results,
                "grid_image_b64": grid_b64,
            },
        }
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Erreur traitement image: {e}")


@app.get("/models/status")
def models_status():
    """Retourne le statut d'entraînement et les métriques de chaque modèle."""
    if engine is None:
        raise HTTPException(503, "Moteur non initialisé")
    return engine.get_status()


@app.get("/generate")
def generate_puzzle(clues: int = 30):
    """Génère un puzzle Sudoku aléatoire avec sa solution."""
    if clues < 17 or clues > 60:
        raise HTTPException(400, "clues doit être entre 17 et 60")
    puzzle, solution = generate_sample(clues)
    return {
        "puzzle": puzzle.tolist(),
        "solution": solution.tolist(),
        "num_clues": int(clues),
        "formatted": format_board(puzzle),
    }


@app.get("/_reload_models")
def reload_models():
    """Recharge les poids après entraînement en arrière-plan."""
    if engine is None:
        return {"reloaded": False}
    engine._load_all_models()
    return {"reloaded": True, "trained": sorted(engine.trained_models)}


@app.post("/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Lance l'entraînement en tâche de fond."""
    def run_training():
        import subprocess
        cmd = [sys.executable, "-m", "training.train_all"]
        if req.quick:
            cmd.append("--quick")
        if req.model:
            cmd += ["--model", req.model]
        subprocess.run(cmd, cwd=str(Path(__file__).parent))
        # Recharger les modèles après entraînement
        engine._load_all_models()

    background_tasks.add_task(run_training)
    return {"message": "Entraînement lancé en arrière-plan", "quick": req.quick}


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
