# neural_network_encryption.py

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkEncryptionProtocol:
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

    def encrypt_message(self, message):
        tensor_message = torch.tensor([ord(c) for c in message])
        output = self.model(tensor_message)
        return output

    def decrypt_message(self, output):
        noise = torch.randn_like(output)
        decrypted_output = output + noise
        return decrypted_output

    def decode_message(self, decrypted_output):
        text_message = ''
        for i in range(decrypted_output.shape[0]):
            text_message += chr(int(decrypted_output[i].item()))
        return text_message
