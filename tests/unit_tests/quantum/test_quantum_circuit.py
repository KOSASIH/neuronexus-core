# test_quantum_circuit.py

import unittest
from quantum.quantum_circuit import QuantumCircuit

class TestQuantumCircuit(unittest.TestCase):
    def test_apply_gate(self):
        qc = QuantumCircuit(2)
        qc.apply_gate('X', 0)
        self.assertEqual(qc.get_state(), [0, 1, 0, 0])

    def test_measure(self):
        qc = QuantumCircuit(2)
        qc.apply_gate('X', 0)
        measurement = qc.measure()
        self.assertIn(measurement, [0, 1])

if __name__ == '__main__':
    unittest.main()
