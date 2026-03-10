import torch
import torch.nn as nn


class MyRNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout=0.2,
        device=None
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):

        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device
        )

        # Forward pass
        out, _ = self.rnn(x, h0)

        # Last timestep
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc(out)

        return out


if __name__ == "__main__":

    input_size = 10
    hidden_size = 64
    num_layers = 2
    output_size = 1
    seq_length = 5

    model = MyRNNModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )

    model.to(model.device)

    x = torch.randn(3, seq_length, input_size).to(model.device)

    output = model(x)

    print("Output shape:", output.shape)