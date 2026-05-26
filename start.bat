@echo off
title Sudoku AI
echo.
echo  Sudoku AI — Lancement automatique
echo  ===================================
echo.
echo  Ce script va :
echo    1. Installer les dependances Python et Node.js
echo    2. Preparer le dataset (auto-detection Kaggle ou generation)
echo    3. Entrainer les 6 modeles de Deep Learning
echo    4. Lancer l'API FastAPI et le frontend Next.js
echo.
echo  Pour un test rapide (2 min) : python run.py --quick
echo  Avec Kaggle CSV             : python run.py --kaggle data/sudoku.csv
echo  Sans reentrain              : python run.py --skip-train
echo.

python run.py %*

pause
