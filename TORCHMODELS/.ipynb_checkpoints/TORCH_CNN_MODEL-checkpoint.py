import torch
import torch.nn as nn
import torch.nn.functional as F

class MY_CNNModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, device=None):
        """
        Args:
            input_channels: Number of input channels (e.g., 3 for RGB images)
            num_classes: Number of output classes. Use 1 for regression or binary classification
            device: torch device (optional, auto-detects if None)
        """
        super(MY_CNNModel, self).__init__()

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # adjust 4*4 if image size differs
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolution + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # raw logits

# Example usage
if __name__ == "__main__":
    # Dummy input: batch_size=3, channels=3, 32x32 images
    x = torch.randn(3, 3, 32, 32)

    # Create CNN model for 10-class classification
    model = MY_CNNModel(input_channels=3, num_classes=10)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    output = model(x)
    print("Output shape:", output.shape)  # Expected: (3, 10)