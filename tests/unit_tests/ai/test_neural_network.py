# test_neural_network.py

import unittest
from ai.neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def test_forward_pass(self):
        nn = NeuralNetwork(784, 256, 10)
        input_data = np.random.rand(1, 784)
        output = nn.forward_pass(input_data)
        self.assertEqual(output.shape, (1, 10))

    def test_backward_pass(self):
        nn = NeuralNetwork(784, 256, 10)
        input_data = np.random.rand(1, 784)
        output = nn.forward_pass(input_data)
        loss = nn.calculate_loss(output, np.random.rand(1, 10))
        self.assertGreater(loss, 0)

if __name__ == '__main__':
    unittest.main()
