import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Decentralized Network(nn.Module):
    def __init__(self, num_nodes, num_edges):
        super(DecentralizedNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.adjacency_matrix = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.edge_weights = nn.Parameter(torch.randn(num_edges, 1))

    def forward(self, input_data):
        output = torch.matmul(self.adjacency_matrix, input_data)
        output = torch.matmul(output, self.edge_weights)
        return output

class BlockchainNetwork(nn.Module):
    def __init__(self, num_nodes, num_blocks):
        super(BlockchainNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.num_blocks = num_blocks
        self.blockchain = nn.Parameter(torch.randn(num_blocks, num_nodes))

    def forward(self, input_data):
        output = torch.matmul(self.blockchain, input_data)
        return output

class PeerToPeerNetwork(nn.Module):
    def __init__(self, num_nodes, num_peers):
        super(PeerToPeerNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.num_peers = num_peers
        self.peer_matrix = nn.Parameter(torch.randn(num_nodes, num_peers))

    def forward(self, input_data):
        output = torch.matmul(self.peer_matrix, input_data)
        return output

# Create an instance of each network
decentralized_network = DecentralizedNetwork(10, 20)
blockchain_network = BlockchainNetwork(10, 20)
peer_to_peer_network = PeerToPeerNetwork(10, 20)

# Train each network
criterion = nn.MSELoss()
optimizer = optim.SGD(decentralized_network.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = decentralized_network(torch.randn(1, 10))
    loss = criterion(output, torch.randn(1, 10))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = blockchain_network(torch.randn(1, 10))
    loss = criterion(output, torch.randn(1, 10))
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = peer_to_peer_network(torch.randn(1, 10))
    loss = criterion(output, torch.randn(1, 10))
    loss.backward()
    optimizer.step()

# Test each network
test_input = torch.randn(1, 10)
print("Decentralized Network Output:", decentralized_network(test_input))
print("Blockchain Network Output:", blockchain_network(test_input))
print("Peer To Peer Network Output:", peer_to_peer_network(test_input))
