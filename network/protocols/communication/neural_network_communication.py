# neural_network_communication.py

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkCommunicationProtocol:
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

    def encode_message(self, message):
        tensor_message = torch.tensor([ord(c) for c in message])
        output = self.model(tensor_message)
        return output

    def transmit_message(self, output):
        noise = torch.randn_like(output)
        transmitted_output = output + noise
        return transmitted_output

    def decode_message(self, transmitted_output):
        decoded_output = self.model(transmitted_output)
        text_message = ''
        for i in range(decoded_output.shape[0]):
            text_message += chr(int(decoded_output[i].item()))
        return text_message
