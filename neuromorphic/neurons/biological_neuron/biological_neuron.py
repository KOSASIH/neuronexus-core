import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BiologicalNeuron(nn.Module):
 def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiologicalNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HodgkinHuxleyNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HodgkinHuxleyNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

class IzhikevichNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IzhikevichNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

class MorrisLecarNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MorrisLecarNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

class FitzHughNagumoNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FitzHughNagumoNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

class HindmarshRoseNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HindmarshRoseNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

class WilsonCowanNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WilsonCowanNeuron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

# Create an instance of each neuron
biological_neuron = BiologicalNeuron(5, 10, 5)
hodgkin_huxley_neuron = HodgkinHuxleyNeuron(5, 10, 5)
izhikevich_neuron = IzhikevichNeuron(5, 10, 5)
morris_lecar_neuron = MorrisLecarNeuron(5, 10, 5)
fitz_hugh_nagumo_neuron = FitzHughNagumoNeuron(5, 10, 5)
hindmarsh_rose_neuron = HindmarshRoseNeuron(5, 10, 5)
wilson_cowan_neuron = WilsonCowanNeuron(5, 10, 5)

# Train each neuron
criterion = nn.MSELoss()
optimizer = optim.SGD(biological_neuron.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = biological_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = hodgkin_huxley_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = izhikevich_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = morris_lecar_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = fitz_hugh_nagumo_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = hindmarsh_rose_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = wilson_cowan_neuron(torch.randn(1, 5))
    loss = criterion(output, torch.randn(1, 5))
    loss.backward()
    optimizer.step()

# Test each neuron
test_input = torch.randn(1, 5)
print("Biological Neuron Output:", biological_neuron(test_input))
print("Hodgkin Huxley Neuron Output:", hodgkin_huxley_neuron(test_input))
print("Izhikevich Neuron Output:", izhikevich_neuron(test_input))
print("Morris Lecar Neuron Output:", morris_lecar_neuron(test_input))
print("FitzHugh Nagumo Neuron Output:", fitz_hugh_nagumo_neuron(test_input))
print("Hindmarsh Rose Neuron Output:", hindmarsh_rose_neuron(test_input))
print("Wilson Cowan Neuron Output:", wilson_cowan_neuron(test_input))
