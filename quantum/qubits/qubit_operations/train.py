import numpy as np
from qubit_operations import QubitOperations

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
