# PyTorch Sequence Models and Trainer

This repository contains **PyTorch implementations** of sequence models (RNN, LSTM, GRU) along with a **flexible Trainer class** for supervised learning tasks. It is designed for regression, binary classification, and multiclass classification on sequential data.

This also highlights the importance of OOP in dealing with Pytorch

---

## Features

- **Sequence Models**
  - `MY_RNNModel`: Vanilla RNN for sequence data.
  - `MY_LSTMModel`: LSTM network with support for multi-layer and sequence-to-one output.
  - `MY_GRUModel`: GRU network with multi-layer support.
  - `MY_CNNModel`: CNN model with multi-class support. 

- **Trainer Class**
  - Handles training loops, metrics, and optimization.
  - Supports automatic task detection:
    - Regression → `MSELoss` + `MeanSquaredError` metric.
    - Binary classification → `BCEWithLogitsLoss` + `Accuracy`.
    - Multiclass classification → `CrossEntropyLoss` + `Accuracy`.
  - Automatically handles device placement (CPU/GPU).
  - Tracks training history (`loss` and `metric`) and provides plotting utilities.

- **Easy Inference**
  - Raw logits are returned from the models.
  - Probabilities can be computed using `torch.sigmoid` (binary) or `torch.softmax` (multiclass) during inference.

---

## Installation and Importing Libraries

```bash
git clone 
pip install torch torchvision torchaudio torchmetrics matplotlib
cd <repository_directory> # You can now import the libraries in your python/jupyter
