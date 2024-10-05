import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Spiking NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.spike_fn = nn.ReLU()
        self.reset_fn = nn.ReLU()

    def forward(self, x):
        x = self.spike_fn(self.fc1(x))
        x = self.reset_fn(self.fc2(x))
        return x

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

class LIFNeuron(nn.Module):
    def __init__(self, tau, v_th, v_reset):
        super(LIFNeuron, self).__init__()
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset

        self.v = torch.zeros(1)

    def forward(self, x):
        dvdt = (self.v - self.v_reset) / self.tau + x
        self.v = self.v + dvdt
        spike = torch.where(self.v >= self.v_th, 1, 0)
        self.v = torch.where(spike, self.v_reset, self.v)
        return spike

class IzhikevichNeuron(nn.Module):
    def __init__(self, a, b, c, d):
        super(IzhikevichNeuron, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.v = torch.zeros(1)
        self.u = torch.zeros(1)

    def forward(self, x):
        dvdt = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + x
        dudt = self.a * (self.b * self.v - self.u)
        self.v = self.v + dvdt
        self.u = self.u + dudt
        spike = torch.where(self.v >= 30, 1, 0)
        self.v = torch.where(spike, self.c, self.v)
        self.u = torch.where(spike, self.u + self.d, self.u)
        return spike

class SpikingNeuralNetworkTrainer:
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

    # Define the spiking neural network model
    model = SpikingNeuralNetwork(input_size=784, hidden_size=256, output_size=10)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Define the trainer
    trainer = SpikingNeuralNetworkTrainer(model, optimizer, loss_fn)

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
