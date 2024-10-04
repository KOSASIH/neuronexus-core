import numpy as np
from qiskit import QuantumCircuit , execute, Aer

class QiskitSimulator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def create_circuit(self):
        circuit = QuantumCircuit(self.n_qubits)
        return circuit

    def add_hadamard(self, circuit, qubit):
        circuit.h(qubit)
        return circuit

    def add_cnot(self, circuit, control, target):
        circuit.cx(control, target)
        return circuit

    def add_measure(self, circuit, qubit):
        circuit.measure(qubit, qubit)
        return circuit

    def simulate(self, circuit):
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit, simulator)
        result = job.result()
        counts = result.get_counts()
        return counts

def main():
    simulator = QiskitSimulator(5)
    circuit = simulator.create_circuit()
    circuit = simulator.add_hadamard(circuit, 0)
    circuit = simulator.add_cnot(circuit, 0, 1)
    circuit = simulator.add_measure(circuit, 1)
    counts = simulator.simulate(circuit)
    print(counts)

if __name__ == "__main__":
    main()
