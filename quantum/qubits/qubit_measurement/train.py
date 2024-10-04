import numpy as np
from qubit_measurement import QubitMeasurement

def main():
    qubit_measurement = QubitMeasurement()
    measure_z_circuit = qubit_measurement.measure_z_basis(0)
    measure_x_circuit = qubit_measurement.measure_x_basis(0)
    measure_y_circuit = qubit_measurement.measure_y_basis(0)

    simulator = Aer.get_backend('qasm_simulator')
    job = execute(measure_z_circuit, simulator)
    result = job.result()
    print(result.get_counts())

    job = execute(measure_x_circuit, simulator)
    result = job.result()
    print(result.get_counts())

    job = execute(measure_y_circuit, simulator)
    result = job.result()
    print(result.get_counts())

if __name__ == "__main__":
    main()
