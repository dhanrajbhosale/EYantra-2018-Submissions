from context import nnet
from nnet import loss

import unittest
import torch
import math
import numpy as np

class TestLossModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests

    def test_cross_entropy(self):
        # settings
        batch_size = 4
        N_out = 10
        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float)
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)

        creloss = loss.cross_entropy_loss(outputs, labels)
        assert type(creloss.item()) == float
        # write more robust and rigourous test cases here
        pass
    
    def test_delta_cross_entropy_loss(self):
        # settings
        batch_size = 4
        N_out = 10
        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float)
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)

        grads_creloss = loss.delta_cross_entropy_softmax(outputs, labels)

        assert isinstance(grads_creloss, torch.FloatTensor)
        assert grads_creloss.size() == torch.Size([batch_size, N_out])
        # write more robust test cases here
        # you should write gradient checking code here
        pass

if __name__ == '__main__':
    unittest.main()