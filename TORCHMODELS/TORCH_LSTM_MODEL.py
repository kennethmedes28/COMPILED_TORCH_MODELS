import torch
import torch.nn as nn

class MY_LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device=None):
        super(MY_LSTMModel, self).__init__()

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
    
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
    
        # Last time step
        out = self.fc(out[:, -1, :])
    
        return out  # raw logits or regression output

# Example usage inside __main__
if __name__ == "__main__":
    # Parameters
    input_size = 10     # number of features per time step
    hidden_size = 50    # number of hidden units
    num_layers = 2      # number of stacked LSTM layers
    output_size = 1     # regression output. Needs modification if classification task
    seq_length = 5      # number of time steps

    # Create model
    model = MY_LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.to(model.device)

    # Dummy input: batch_size=3, seq_length=5, input_size=10
    x = torch.randn(3, seq_length, input_size).to(model.device)

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)