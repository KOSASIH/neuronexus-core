import numpy as np
from qiskit import QuantumCircuit, execute, Aer

class QubitOperations:
    def __init__(self):
        pass

    def hadamard_gate(self, qubit):
        circuit = QuantumCircuit(1)
        circuit.h(qubit)
        return circuit

    def pauli_x_gate(self, qubit):
        circuit = QuantumCircuit(1)
        circuit.x(qubit)
        return circuit

    def pauli_y_gate(self, qubit):
        circuit = QuantumCircuit(1)
        circuit.y(qubit)
        return circuit

    def pauli_z_gate(self, qubit):
        circuit = QuantumCircuit(1)
        circuit.z(qubit)
        return circuit

    def cnot_gate(self, control_qubit, target_qubit):
        circuit = QuantumCircuit(2)
        circuit.cx(control_qubit, target_qubit)
        return circuit

    def swap_gate(self, qubit1, qubit2):
        circuit = QuantumCircuit(2)
        circuit.swap(qubit1, qubit2)
        return circuit

def main():
    qubit_operations = QubitOperations()
    hadamard_circuit = qubit_operations.hadamard_gate(0)
    pauli_x_circuit = qubit_operations.pauli_x_gate(0)
    pauli_y_circuit = qubit_operations.pauli_y_gate(0)
    pauli_z_circuit = qubit_operations.pauli_z_gate(0)
    cnot_circuit = qubit_operations.cnot_gate(0, 1)
    swap_circuit = qubit_operations.swap_gate(0, 1)

    simulator = Aer.get_backend('qasm_simulator')
    job = execute(hadamard_circuit, simulator)
    result = job.result()
    print(result.get_counts())

    job = execute(pauli_x_circuit, simulator)
    result = job.result()
    print(result.get_counts())

    job = execute(pauli_y_circuit, simulator)
    result = job.result()
    print(result.get_counts())

    job = execute(pauli_z_circuit, simulator)
    result = job.result()
    print(result.get_counts())

    job = execute(cnot_circuit, simulator)
    result = job.result()
    print(result.get_counts())

    job = execute(swap_circuit, simulator)
    result = job.result()
    print(result.get_counts())

if __name__ == "__main__":
    main()
