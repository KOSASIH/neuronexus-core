import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ArtificialNeuron(nn .Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ArtificialNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReLUNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReLUNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class SigmoidalNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SigmoidalNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class SoftmaxNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SoftmaxNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.softmax(self.fc1(x), dim=1)
        x = torch.softmax(self.fc2(x), dim=1)
        return x

class ParametricReLUNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ParametricReLUNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.alpha * x + (1 - self.alpha) * torch.relu(self.fc2(x))
        return x

class ParametricLeakyReLUNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ParametricLeakyReLUNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.alpha * x + (1 - self.alpha) * torch.relu(self.fc2(x))
        return x

class SwishNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SwishNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x * torch.sigmoid(x)
        x = self.fc1(x)
        x = x * torch.sigmoid(x)
        x = self.fc2(x)
        return x

class GELUNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GELUNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
        x = self.fc1(x)
        x = 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
        x = self.fc2(x)
        return x

class SoftClippingNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SoftClippingNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.clamp(x, min=-1, max=1)
        x = self.fc1(x)
        x = torch.clamp(x, min=-1, max=1)
        x = self.fc2(x)
        return x

# Create an instance of each neuron
relu_neuron = ReLUNeuron(5, 10, 5)
sigmoidal_neuron = SigmoidalNeuron(5, 10, 5)
softmax_neuron = SoftmaxNeuron(5, 10, 5)
parametric_relu_neuron = ParametricReLUNeuron(5, 10, 5)
parametric_leaky_relu_neuron = ParametricLeakyReLUNeuron(5, 10, 5)
swish_neuron = SwishNeuron(5, 10, 5)
gelu_neuron = GELUNeuron(5, 10, 5)
softclipping_neuron = SoftClippingNeuron(5, 10, 5)

# Train each neuron
criterion = nn.MSELoss()
optimizer = optim.SGD(relu_neuron.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = relu_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = sigmoidal_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = softmax_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = parametric_relu_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = parametric_leaky_relu_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = swish_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = gelu_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = softclipping_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()
