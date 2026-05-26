==============================================================
  SUDOKU AI — Dossier data/
==============================================================

Ce dossier contient les données d'entraînement.
Il est vide par défaut (les fichiers sont trop lourds pour git).

--------------------------------------------------------------
OPTION A — Dataset Kaggle (RECOMMANDÉ)
--------------------------------------------------------------
  Source  : https://www.kaggle.com/datasets/bryanpark/sudoku
  Fichier : sudoku.csv  (~113 MB, 1 000 000 puzzles)
  Format  : CSV avec colonnes 'quizzes' et 'solutions'
            chaque valeur = chaîne de 81 chiffres (0=vide)

  Étapes :
    1. Créer un compte Kaggle (gratuit)
    2. Télécharger sudoku.csv
    3. Placer le fichier ici : backend/data/sudoku.csv
    4. Lancer : python -m training.data_pipeline --kaggle data/sudoku.csv

  Sur Colab/Kaggle (avec API key) :
    !kaggle datasets download -d bryanpark/sudoku -p data/ --unzip
    python -m training.data_pipeline --kaggle data/sudoku.csv

--------------------------------------------------------------
OPTION B — Génération automatique (aucun téléchargement)
--------------------------------------------------------------
  Génère les puzzles par backtracking (plus lent mais autonome).

    python -m training.data_pipeline --generate
    python -m training.data_pipeline --generate --train 200000

--------------------------------------------------------------
FICHIERS PRODUITS (après data_pipeline)
--------------------------------------------------------------
  data/
  ├── sudoku.csv              ← Source Kaggle (à télécharger)
  ├── cleaned.npz             ← Données nettoyées (puzzles+solutions)
  ├── train.npz               ← Set d'entraînement
  ├── val.npz                 ← Set de validation
  ├── test.npz                ← Set de test (isolé)
  └── stats.json              ← Statistiques du dataset

--------------------------------------------------------------
STRUCTURE INTERNE DES FICHIERS .npz
--------------------------------------------------------------
  puzzles   : (N, 81) int8  — 0=vide, 1-9=indice donné
  solutions : (N, 81) int8  — 1-9 (grille complète)
  clues     : (N,)    int8  — nombre d'indices par puzzle
  difficulty: (N,)    int8  — 0=easy,1=medium,2=hard,3=expert
==============================================================
