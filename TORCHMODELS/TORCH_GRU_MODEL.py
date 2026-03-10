import torch
import torch.nn as nn

class MY_GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device=None):
        super(MY_GRUModel, self).__init__()

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # Take the output from the last time step
        out = self.fc(out[:, -1, :])

        return out  # raw logits or regression output


# Example usage
if __name__ == "__main__":
    # Parameters
    input_size = 10     # number of features per time step
    hidden_size = 50    # number of hidden units
    num_layers = 2      # number of stacked GRU layers
    output_size = 1     # regression output. Change for classification tasks
    seq_length = 5      # number of time steps

    # Create model
    model = MY_GRUModel(input_size, hidden_size, num_layers, output_size)
    model.to(model.device)

    # Dummy input: batch_size=3, seq_length=5, input_size=10
    x = torch.randn(3, seq_length, input_size).to(model.device)

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)