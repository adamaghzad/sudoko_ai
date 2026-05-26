"""
Raccourci : entraîne tous les modèles en mode rapide (10k puzzles, 5 époques).
Usage : python train_quick.py
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "training.train_all", "--quick"], check=True)
