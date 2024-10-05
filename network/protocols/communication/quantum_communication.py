# quantum_communication.py

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import Statevector

class QuantumCommunicationProtocol:
    def __init__(self, num_qubits, error_correction=True):
        self.num_qubits = num_qubits
        self.error_correction = error_correction
        self.qc = QuantumCircuit(num_qubits)

    def encode_message(self, message):
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        self.qc.x(0)  # Initialize the first qubit to |1
        for i, bit in enumerate(binary_message):
            if bit == '1':
                self.qc.x(i+1)  # Apply X gate to encode the bit
        if self.error_correction:
            self.qc.barrier()
            self.qc.cx(0, 1)  # Apply CNOT gate to encode the parity bit
            self.qc.cx(0, 2)  # Apply CNOT gate to encode the parity bit

    def transmit_message(self):
        job = execute(self.qc, backend='qasm_simulator')
        result = job.result()
        counts = result.get_counts()
        message = ''
        for i in range(self.num_qubits):
            if counts.get('1' * (i+1) + '0' * (self.num_qubits - i - 1), 0) > 0:
                message += '1'
            else:
                message += '0'
        return message

    def decode_message(self, message):
        text_message = ''
        for i in range(0, len(message), 8):
            byte = message[i:i+8]
            text_message += chr(int(byte, 2))
        return text_message
