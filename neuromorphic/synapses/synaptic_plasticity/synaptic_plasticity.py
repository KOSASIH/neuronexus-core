import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SynapticPlasticity(nn.Module):
    def __init__(self, input_size, output_size):
        super(SynapticPlasticity, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, output_size)

        self.synaptic_weights = nn.Parameter(torch.randn(input_size, output_size))

    def forward(self, x):
        output = torch.matmul(x, self.synaptic_weights)
        return output

    def update_synaptic_weights(self, x, y, learning_rate):
        dw = torch.matmul(x.T, (y - torch.matmul(x, self.synaptic_weights)))
        self.synaptic_weights.data += learning_rate * dw

class HebbianLearning(nn.Module):
    def __init__(self, input_size, output_size):
        super(HebbianLearning, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, output_size)

        self.synaptic_weights = nn.Parameter(torch.randn(input_size, output_size))

    def forward(self, x):
        output = torch.matmul(x, self.synaptic_weights)
        return output

    def update_synaptic_weights(self, x, y, learning_rate):
        dw = torch.matmul(x.T, y)
        self.synaptic_weights.data += learning_rate * dw

class SpikeTimingDependentPlasticity(nn.Module):
    def __init__(self, input_size, output_size):
        super(SpikeTimingDependentPlasticity, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, output_size)

        self.synaptic_weights = nn.Parameter(torch.randn(input_size, output_size))

    def forward(self, x):
        output = torch.matmul(x, self.synaptic_weights)
        return output

    def update_synaptic_weights(self, x, y, learning_rate, tau):
        dw = torch.matmul(x.T, (y - torch.matmul(x, self.synaptic_weights)))
        dw = dw * torch.exp(-torch.abs(y - torch.matmul(x, self.synaptic_weights)) / tau)
        self.synaptic_weights.data += learning_rate * dw

class HomeostaticPlasticity(nn.Module):
    def __init__(self, input_size, output_size):
        super(HomeostaticPlasticity, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, output_size)

        self.synaptic_weights = nn.Parameter(torch.randn(input_size, output_size))

    def forward(self, x):
        output = torch.matmul(x, self.synaptic_weights)
        return output

    def update_synaptic_weights(self, x, y, learning_rate, target_rate):
        dw = torch.matmul(x.T, (y - target_rate))
        self.synaptic_weights.data += learning_rate * dw

class SynapticPlasticityTrainer:
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

    # Define the synaptic plasticity model
    model = SynapticPlasticity(input_size=784, output_size=10)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Define the trainer
    trainer = SynapticPlasticityTrainer(model, optimizer, loss_fn)

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
