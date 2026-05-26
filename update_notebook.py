import json

with open("e:/sudoko_ai/notebooks/colab_training.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        # Check if this is the dataset config cell
        if "USE_KAGGLE = False   # ← True pour " in "".join(cell.get("source", [])):
            cell["source"] = [
                "# ══════════════════════════════════════════════════════════\n",
                "# CHOIX DU DATASET — modifiez ici\n",
                "# ══════════════════════════════════════════════════════════\n",
                "USE_KAGGLE = True   # ← True pour télécharger 1M puzzles depuis Kaggle\n",
                "\n",
                "# Tailles du dataset (adaptez selon votre runtime)\n",
                "#  T4 Colab  : TRAIN_SIZE = 500_000  (~45 min / modèle)\n",
                "#  P100 Kaggle: TRAIN_SIZE = 1_000_000\n",
                "#  Test rapide: TRAIN_SIZE = 50_000  (~5 min / modèle)\n",
                "TRAIN_SIZE = 500_000\n",
                "VAL_SIZE   =  10_000\n",
                "TEST_SIZE  =   5_000\n",
                "BATCH_SIZE = 1024\n",
                "# ══════════════════════════════════════════════════════════\n",
                "\n",
                "KAGGLE_CSV = None\n",
                "\n",
                "if USE_KAGGLE:\n",
                "    try:\n",
                "        import subprocess\n",
                "        subprocess.run(['pip', 'install', 'kagglehub', '-q'], check=False)\n",
                "        import kagglehub\n",
                "        from pathlib import Path\n",
                "        print('Téléchargement Kaggle bryanpark/sudoku (~113 MB) via kagglehub...')\n",
                "        path = kagglehub.dataset_download(\"bryanpark/sudoku\")\n",
                "        csv_candidate = Path(path) / 'sudoku.csv'\n",
                "        if csv_candidate.exists():\n",
                "            KAGGLE_CSV = str(csv_candidate)\n",
                "            print(f'✓ Dataset Kaggle téléchargé : {KAGGLE_CSV}')\n",
                "        else:\n",
                "            print(f'✗ Fichier sudoku.csv introuvable dans {path}')\n",
                "            print('  → Basculement en mode génération')\n",
                "    except Exception as e:\n",
                "        print(f'✗ Erreur kagglehub : {e}')\n",
                "        print('  → Basculement en mode génération')\n",
                "\n",
                "if KAGGLE_CSV:\n",
                "    print(f'\\nSource : Kaggle CSV ({KAGGLE_CSV})')\n",
                "else:\n",
                "    print('\\nSource : génération par backtracking')\n"
            ]
            break

with open("e:/sudoko_ai/notebooks/colab_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    # The original file might have a trailing newline
    f.write('\n')

print("Notebook updated successfully.")
