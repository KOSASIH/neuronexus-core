import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpikingNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.spike_fn = nn.ReLU()

    def forward(self, x):
        x = self.spike_fn(self.fc1(x))
        x = self.fc2(x)
        return x

    def spike(self, x):
        return self.spike_fn(x)

class SpikingNeuron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpikingNeuron, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(input_dim, output_dim)

        self.spike_fn = nn.ReLU()

    def forward(self, x):
        x = self.spike_fn(self.fc(x))
        return x

    def spike(self, x):
        return self.spike_fn(x)

def train_snn(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_snn(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Define the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model, loss function, and optimizer
    model = SpikingNeuralNetwork(input_dim=784, hidden_dim=256, output_dim=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the data loader
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Train the model
    for epoch in range(10):
        train_snn(model, device, trainloader, optimizer, criterion)
        test_snn(model, device, testloader)