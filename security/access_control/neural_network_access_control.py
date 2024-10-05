# neural_network_access_control.py

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkAccessControlProtocol:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def grant_access(self, user_id):
        tensor_user_id = torch.tensor([ord(c) for c in user_id])
        output = self.model(tensor_user_id)
        return output

    def deny_access(self, output):
        noise = torch.randn_like(output)
        denied_output = output + noise
        return denied_output

    def decode_access(self, denied_output):
        if denied_output.item() > 0.5:
            return "Access granted"
        else:
            return "Access denied"
