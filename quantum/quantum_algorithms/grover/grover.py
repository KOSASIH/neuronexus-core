import numpy as np
from qiskit import QuantumCircuit, execute, Aer

class Grover:
    def __init__(self, n):
        self.n = n

    def grover(self):
        # Create a quantum circuit with n qubits
        circuit = QuantumCircuit(self.n)

        # Apply Hadamard gates to all qubits
        for i in range(self.n):
            circuit.h(i)

        # Apply the Grover iteration
        for _ in range(int(np.sqrt(2 ** self.n))):
            # Apply the oracle
            circuit.x(self.n - 1)
            circuit.barrier()
            # Apply the diffusion operator
            circuit.h(self.n - 1)
            circuit.x(self.n - 1)
            circuit.h(self.n - 1)
            circuit.barrier()

        # Measure the qubits
        circuit.measure_all()

        # Execute the circuit
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit, simulator)
        result = job.result()
        counts = result.get_counts()

        # Find the most likely outcome
        outcome = max(counts, key=counts.get)

        # Return the outcome
        return outcome

def main():
    grover = Grover(5)
    outcome = grover.grover()
    print(outcome)

if __name__ == "__main__":
    main()
