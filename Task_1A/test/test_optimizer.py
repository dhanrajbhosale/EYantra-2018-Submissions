from context import nnet
from nnet import optimizer

import unittest
import torch
import math
import numpy as np

class TestOptimizerModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests

    def test_mbgd(self):
        N_in = 28 * 28
        N_h1 = 512
        N_h2 = 256
        N_out = 10

        w1 = torch.randn((N_h1, N_in), dtype=torch.float)
        w2 = torch.randn((N_h2, N_h1), dtype=torch.float)
        w3 = torch.randn((N_out, N_h2), dtype=torch.float)
        b1 = torch.randn(N_h1, dtype=torch.float)
        b2 = torch.randn(N_h2, dtype=torch.float)
        b3 = torch.randn(N_out, dtype=torch.float)
        
        weights = {'w1': w1, 'w2': w2, 'w3': w3} 
        biases =  {'b1': b1, 'b2': b2, 'b3': b3}
        all_params = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3}

        dw1 = torch.randn_like(w1)
        db1 = torch.randn_like(b1)
        dw2 = torch.randn_like(w2)
        db2 = torch.randn_like(b2)
        dw3 = torch.randn_like(w3)
        db3 = torch.randn_like(b3)
        
        lr = 0.001

        n_weights, n_biases = optimizer.mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr)
        assert isinstance(n_weights, dict)
        assert isinstance(n_biases, dict)
        
        for params in [n_weights, n_biases]:
            for key, value in params.items():
                assert isinstance(value, torch.FloatTensor)
                assert params[key].size() == all_params[key].size()



if __name__ == '__main__':
    unittest.main()
