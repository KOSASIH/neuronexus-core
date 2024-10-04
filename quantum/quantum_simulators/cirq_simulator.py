import numpy as np
import cirq

class CirqSimulator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def create_circuit(self):
        qubits = [cirq.LineQubit(i) for i in range(self.n_qubits)]
        circuit = cirq.Circuit()
        return circuit, qubits

    def add_hadamard(self, circuit, qubit):
        circuit.append(cirq.H(qubit))
        return circuit

    def add_cnot(self, circuit, control, target):
        circuit.append(cirq.CNOT(control, target))
        return circuit

    def add_measure(self, circuit, qubit):
        circuit.append(cirq.measure(qubit))
        return circuit

    def simulate(self, circuit):
        simulator = cirq.Simulator()
        result = simulator.run(circuit)
        return result

def main():
    simulator = CirqSimulator(5)
    circuit, qubits = simulator.create_circuit()
    circuit = simulator.add_hadamard(circuit, qubits[0])
    circuit = simulator.add_cnot(circuit, qubits[0], qubits[1])
    circuit = simulator.add_measure(circuit, qubits[1])
    result = simulator.simulate(circuit)
    print(result)

if __name__ == "__main__":
    main()
