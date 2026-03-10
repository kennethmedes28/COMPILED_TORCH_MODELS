import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, MeanSquaredError
import matplotlib.pyplot as plt

class Metrics(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class Trainer:
    def __init__(self, model, lr=0.001, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Auto loss + metric
        if model.output_size == 1:
            self.task = "regression"
            self.criterion = nn.MSELoss()
            self.metric = MeanSquaredError().to(self.device)
            self.metric_name = "MSE"

        elif model.output_size == 2:
            self.task = "binary"
            self.criterion = nn.BCEWithLogitsLoss()
            self.metric = Accuracy(task="binary").to(self.device)
            self.metric_name = "Accuracy"

        else:
            self.task = "multiclass"
            self.criterion = nn.CrossEntropyLoss()
            self.metric = Accuracy(task="multiclass", num_classes=model.output_size).to(self.device)
            self.metric_name = "Accuracy"

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history = {"loss": [], "metric": []}

    def train(self, dataloader, epochs=5):
        """Train the model for given epochs"""
        for epoch in range(epochs):
            self.model.train()
            self.metric.reset()
            epoch_loss = 0

            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)

                # Loss and predictions
                if self.task == "binary":
                    y_true = y.float().unsqueeze(1)
                    loss = self.criterion(outputs, y_true)
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                elif self.task == "multiclass":
                    loss = self.criterion(outputs, y)
                    preds = torch.argmax(outputs, dim=1)
                else:  # regression
                    loss = self.criterion(outputs, y)
                    preds = outputs

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update metric
                self.metric.update(preds, y)
                epoch_loss += loss.item()

            # Average metrics per epoch
            avg_loss = epoch_loss / len(dataloader)
            metric_val = self.metric.compute().item()

            self.history["loss"].append(avg_loss)
            self.history["metric"].append(metric_val)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"{self.metric_name}: {metric_val:.4f}"
            )

    def evaluate(self, dataloader):
        """Evaluate model without gradient tracking"""
        self.model.eval()
        self.metric.reset()
        total_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)

                if self.task == "binary":
                    y_true = y.float().unsqueeze(1)
                    loss = self.criterion(outputs, y_true)
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                elif self.task == "multiclass":
                    loss = self.criterion(outputs, y)
                    preds = torch.argmax(outputs, dim=1)
                else:  # regression
                    loss = self.criterion(outputs, y)
                    preds = outputs

                self.metric.update(preds, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        metric_val = self.metric.compute().item()

        print(f"Evaluation | Loss: {avg_loss:.4f} | {self.metric_name}: {metric_val:.4f}")
        return avg_loss, metric_val

    def plot_metrics(self):
        """Plot training loss and metric"""
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["loss"], marker='o')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.history["metric"], marker='o', color='green')
        plt.title(self.metric_name)
        plt.xlabel("Epoch")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model = Metrics(input_size=10, hidden_size=32, output_size=2)
    trainer = Trainer(model)

    trainer.train(dataloader, epochs=5)
    trainer.evaluate(dataloader)
    trainer.plot_metrics()