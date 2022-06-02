from context import nnet
from nnet import activation
import unittest
import torch
import math
import numpy as np

class TestActivationModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests

    def test_sigmoid(self):
        # This is an example test case. We encourage you to write more such test cases.
        # You can test your code unit-wise (functions, classes, etc.) 
        x = torch.FloatTensor([[-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50], [-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50]])
        y = torch.FloatTensor([[4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999], 
                                [4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999]])
        precision = 0.0009
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(x) - y), precision).all())

    def test_delta_sigmoid(self):
        batch_size = 6
        N_hn = 512

        x = torch.rand((batch_size, N_hn), dtype=torch.float)
        grads = activation.delta_sigmoid(x)
        assert isinstance(grads, torch.FloatTensor)
        assert grads.size() == torch.Size([batch_size, N_hn])

    def test_softmax(self):
        batch_size = 7
        N_out = 10

        x = torch.rand((batch_size, N_out), dtype=torch.float)
        outputs = activation.softmax(x)

        assert isinstance(outputs, torch.FloatTensor)
        assert outputs.size() == torch.Size([batch_size, N_out])
        

if __name__ == '__main__':
    unittest.main()