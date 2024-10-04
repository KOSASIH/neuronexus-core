import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import NeuralNetwork, train, test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_dim=784, hidden_dim=256, output_dim=10)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = ...
    test_loader = ...
    for epoch in range(10):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        test_loss = test(model, device, test_loader, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
