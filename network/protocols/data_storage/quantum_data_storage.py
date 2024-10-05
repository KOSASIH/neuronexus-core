# quantum_data_storage.py

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import Statevector

class QuantumDataStorageProtocol:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qc = QuantumCircuit(num_qubits)

    def store_data(self, data):
        binary_data = ''.join(format(ord(c), '08b') for c in data)
        self.qc.x(0)  # Initialize the first qubit to |1
        for i, bit in enumerate(binary_data):
            if bit == '1':
                self.qc.x(i+1)  # Apply X gate to store the bit

    def retrieve_data(self):
        job = execute(self.qc, backend='qasm_simulator')
        result = job.result()
        counts = result.get_counts()
        data = ''
        for i in range(self.num_qubits):
            if counts.get('1' * (i+1) + '0' * (self.num_qubits - i - 1), 0) > 0:
                data += '1'
            else:
                data += '0'
        return data

    def decode_data(self, data):
        text_data = ''
        for i in range(0, len(data), 8):
            byte = data[i:i+8]
            text_data += chr(int(byte, 2))
        return text_data
