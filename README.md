# Sudoku AI Deep Learning Solver

PyTorch research project for solving Sudoku puzzles using Artificial Intelligence. The study compares exactly six deep learning model architectures:

- **MLP** on flattened tabular grid data;
- **CNN 2D** utilizing spatial inductive bias;
- **Simple RNN** on bidirectional sequences;
- **LSTM** analyzing long-term dependencies with its 3-gate structure;
- **GRU** leveraging a lighter 2-gate recurrent compromise;
- **Hybrid (CNN + LSTM)** combining 2D feature extraction and sequential logic.

This repository serves as an academic and research demonstration of neural network capabilities on strict combinatorial logic problems.

## Data Protocol

The models do not rely on a static external dataset. Instead, a dynamic pipeline generates infinite unique configurations.

- **Generation**: A backtracking algorithm generates solved $9 \times 9$ grids and sequentially removes numbers to reach target clue counts (e.g., 17 to 50 clues), always validating the uniqueness of the solution.
- **Augmentation**: To prevent overfitting and encourage geometric generalization, valid grids are augmented using the Dihedral Group D4 (orthogonal rotations of 90°, 180°, 270° and horizontal/vertical/diagonal reflections).
- **Vision Pipeline**: Real-world images of Sudokus are processed via a CV pipeline involving grayscale conversion, adaptive thresholding, continuous contour detection, warp perspective transforms, and digit extraction via a specialized `DigitCNN`.

## Models & Optimization

### MLP
The Multi-Layer Perceptron acts as the baseline. It projects the $9 \times 9$ grid into an 81-dimensional vector, operating without spatial inductive biases. 

### CNN
The Convolutional Neural Network processes the 2D grid structure, using kernels to natively understand local relationships (rows, columns, and $3 \times 3$ subgrids).

### Recurrent Models (RNN, LSTM, GRU)
These sequence models read the Sudoku cell by cell. 
- The standard **RNN** struggles with vanishing gradients over the 81 steps.
- The **LSTM** successfully captures long dependencies (correlations between distant cells) storing spatial logic in its hidden cell states.
- The **GRU** achieves comparable performance to LSTM with a condensed parametric structure and faster inference.

### Hybrid (CNN-LSTM)
Extracts high-density spatial feature maps via convolutions and sequentially processes them through LSTM layers to resolve overarching directional logic.

### Optimization: ConfidenceAwareLoss
Instead of standard Categorical Cross-Entropy, the models are trained using a custom `ConfidenceAwareLoss`. This heavily penalizes the network for uncertainty on easily deducible empty cells, forcing binary-like, strict classification and avoiding probability over-smoothing.

## Repository Layout
```text
.
|-- backend/
|   |-- models/             # PyTorch implementations (MLP, CNN, LSTM...)
|   |-- api.py              # FastAPI server
|   |-- solver.py           # Backtracking and inference logic
|   |-- vision.py           # OpenCV processing and DigitCNN
|   `-- weights/            # Trained weights (*.pt)
|-- frontend/               # UI components and React/Vue views
|-- notebooks/
|   |-- 01_data_generation.ipynb
|   |-- 02_d4_symmetry_augmentation.ipynb
|   |-- 03_training_mlp_cnn.ipynb
|   |-- 04_training_rnn_lstm_gru.ipynb
|   |-- 05_confidence_aware_loss.ipynb
|   `-- archive/
|-- rapport_soutenance_sudoku.tex  # Academic LaTeX Report
|-- run.py                  # Main runner script
|-- start.bat               # Windows launcher script
|-- start.sh                # Unix launcher script
|-- requirements.txt
|-- docker-compose.yml
`-- README.md
```

## Dashboard and Interface
The project incorporates a fully functional UI demonstrating real-time inference:
1. **Random Generation**: Slider-based grid generation.
2. **Computer Vision**: Drag-and-drop photo resolution.
3. **Manual Input**: Interactive visual pad.
4. **Real-Time Monitoring**: Comparative cards showing inference time (ms), mean confidence, and min/max empty cell confidence for all 6 models simultaneously.

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# Install requirements
python -m pip install -r requirements.txt
# Ensure CUDA-compatible PyTorch is installed if running on GPU

# Run the full application
python run.py
```

## Generated Artifacts
Trained weights and model artifacts are saved automatically to track epochs and size:
- `backend/weights/mlp.pt`, `cnn.pt`, `lstm.pt`, etc.
- Training histories in `.json` format

## Metrics
Evaluation provides instantaneous feedback on:
- **Inference Time**: Benchmark of architectural complexity.
- **Mean Confidence**: Global softmax certainty.
- **Empty Cells Min/Max**: Localization of the model's structural hesitation.

The following values reflect the latest evaluation metrics found in the local generated artifacts (`backend/weights/*_history.json`):

| Model | Best Validation Loss | Cell Accuracy | Puzzle Accuracy |
| :--- | :--- | :--- | :--- |
| **MLP** | 1.6684 | 39.40% | 0.00% |
| **RNN** | 0.4105 | 82.93% | 0.00% |
| **GRU** | 0.4133 | 81.81% | 0.00% |
| **LSTM** | 0.4338 | 80.21% | 0.00% |
| **CNN 2D** | 0.2392 | 90.14% | 0.09% |
| **Hybrid** | **0.1236** | **95.47%** | **7.39%** |

*(Note: "Cell Accuracy" measures the model's precision on individual grid positions, while "Puzzle Accuracy" requires all 81 cells to be strictly correct. The protocol winner is the Hybrid CNN-LSTM architecture based on the lowest validation loss and highest accuracy).*

---
*This repository is an academic demonstration in Deep Learning. Model feature maps describe learned heuristics and should not be confused with deterministic solver paths.*
