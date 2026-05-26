import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
GRAD_CLIP = 1.0
NUM_WORKERS = 0

# Data generation
TRAIN_SIZE = 100_000
VAL_SIZE = 10_000
TEST_SIZE = 5_000
MIN_CLUES = 25
MAX_CLUES = 36

# Models
MODEL_NAMES = ["mlp", "cnn", "rnn", "lstm", "gru", "hybrid"]

# Image processing
CELL_SIZE = 28
GRID_SIZE = 252  # 28 * 9

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3001", "http://127.0.0.1:3001"]
