# test_quantum_simulation.py

import unittest
from quantum.quantum_simulation import QuantumSimulation

class TestQuantumSimulation(unittest.TestCase):
    def test_simulate(self):
        qs = QuantumSimulation(2)
        qs.simulate()
        self.assertEqual(qs.get_state(), [0, 1, 0, 0])

    def test_measure(self):
        qs = QuantumSimulation(2)
        qs.simulate()
        measurement = qs.measure()
        self.assertIn(measurement, [0, 1])

if __name__ == '__main__':
    unittest.main()
