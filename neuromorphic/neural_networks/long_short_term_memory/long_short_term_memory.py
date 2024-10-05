import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LongShortTermMemory(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LongShortTermMemory, self).__init__()
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

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc_i = nn.Linear(input_size, hidden_size)
        self.fc_f = nn.Linear(input_size, hidden_size)
        self.fc_g = nn.Linear(input_size, hidden_size)
        self.fc_o = nn.Linear(input_size, hidden_size)

    def forward(self, x, h, c):
        i = torch.sigmoid(self.fc_i(x))
        f = torch.sigmoid(self.fc_f(x))
        g = torch.tanh(self.fc_g(x))
        o = torch.sigmoid(self.fc_o(x))

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        for i in range(x.size(1)):
            h, c = self.lstm_cell(x[:, i, :], h, c)

        out = self.fc(h)
        return out

class LongShortTermMemoryTrainer:
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

    # Define the long short-term memory model
    model = LongShortTermMemory(input_size=784, hidden_size=256, output_size=10)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Define the trainer
    trainer = LongShortTermMemoryTrainer(model, optimizer, loss_fn)

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
