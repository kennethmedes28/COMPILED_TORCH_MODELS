# Sample Usage

import torch
from TORCH_LSTM_MODEL import MY_LSTMModel  # or MY_GRUModel / MY_RNNModel
from TORCH_REGRESSION-CLASSIFICATION_TRAINER import Trainer
from torch.utils.data import DataLoader, TensorDataset

# Dummy sequential data
X = torch.randn(100, 5, 10)  # (batch, seq_length, features)
y = torch.randint(0, 2, (100,))  # binary classification

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create model
model = MY_LSTMModel(input_size=10, hidden_size=50, num_layers=2, output_size=2)
model.to(model.device)

# Create trainer
trainer = Trainer(model)

# Train
trainer.train(dataloader, epochs=5)

# Plot metrics
trainer.plot_metrics()