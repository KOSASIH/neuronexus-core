import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RecurrentNe uralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

class LSTMNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

class GRUNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

class RecurrentNeuralNetworkTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss.item()

# Example usage:
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define the recurrent neural network model
    model = RecurrentNeuralNetwork(input_size=784, hidden_size=256, output_size=10)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Define the trainer
    trainer = RecurrentNeuralNetworkTrainer(model, optimizer, loss_fn)

    # Train the model
    inputs = torch.randn(100, 784)
    targets = torch.randint(0, 10, (100,))
    for epoch in range(10):
        loss = trainer.train(inputs, targets)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # Test the model
    test_inputs = torch.randn(100, 784)
    test_targets = torch.randint(0, 10, (100,))
    test_loss = trainer.test(test_inputs, test_targets)
    print(f"Test Loss: {test_loss:.4f}")
