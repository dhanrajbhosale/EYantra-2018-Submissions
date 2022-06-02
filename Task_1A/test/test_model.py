from context import nnet
from nnet import model, loss, activation, optimizer

import unittest
import torch
import math
import numpy as np

torch.set_printoptions(precision=7)

class TestModelModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests

    def setUp(self):
        # Layer size
        self.label=7
        self.n_in = 28 * 28 # Input size
        self.n_h1 = 1000 # Hidden Layer 1 size
        self.n_h2 = 1000 # Hidden Layer 2 size
        self.n_out = 10 # Output size
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = model.FullyConnected(self.n_in, self.n_h1, self.n_h2, self.n_out, device=self.device)

    def test_init(self):
        assert isinstance(self.model.weights['w1'], torch.FloatTensor)
        assert self.model.weights['w1'].size() == torch.Size([self.n_in,self.n_h1])

        assert isinstance(self.model.weights['w2'], torch.FloatTensor)
        assert self.model.weights['w2'].size() == torch.Size([self.n_h1,self.n_h2])

        assert isinstance(self.model.weights['w3'], torch.FloatTensor)
        assert self.model.weights['w3'].size() == torch.Size([self.n_h2,self.n_out])

        assert isinstance(self.model.biases['b1'], torch.FloatTensor)
        assert self.model.biases['b1'].size() == torch.Size([self.n_h1])

        assert isinstance(self.model.biases['b2'], torch.FloatTensor)
        assert self.model.biases['b2'].size() == torch.Size([self.n_h2])

        assert isinstance(self.model.biases['b3'], torch.FloatTensor)
        assert self.model.biases['b3'].size() == torch.Size([self.n_out])

    def test_train(self):
        # settings
        batch_size = 7
        # tensors
        inputs = torch.randn((batch_size, self.n_in), dtype=torch.float)
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)

        creloss, accuracy, outputs = self.model.train(inputs, labels)

        assert isinstance(creloss, float)
        assert isinstance(accuracy, float)
        assert isinstance(outputs, torch.FloatTensor)
        assert outputs.size() == torch.Size([batch_size, self.n_out])

    def test_predict(self):
        # settings
        batch_size = 7
        # tensors
        inputs = torch.randn((batch_size, self.n_in), dtype=torch.float)
        score, idx = self.model.predict(inputs)
        assert isinstance(score, torch.FloatTensor)
        assert score.size() == torch.Size([batch_size])
        assert isinstance(idx, torch.LongTensor)
        assert idx.size() == torch.Size([batch_size])

    def test_eval(self):
        # settings
        batch_size = 7
        # tensors
        inputs = torch.randn((batch_size, self.n_in), dtype=torch.float)
        
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)
        loss, accuracy, outputs = self.model.eval(inputs, labels)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert isinstance(outputs, torch.FloatTensor)
        assert outputs.size() == torch.Size([batch_size, self.n_out])

    def test_accuracy(self):
        # write test cases like these
        a = torch.FloatTensor([[0.1, 0.2, 0.113, 0.452, 0.978],
                                [0.9, 0.5, 0.2, 0.455, 0.7],
                                [0.1, 0.5, 0.2, 0.455, 0.7]])
        b = torch.tensor([4, 1, 4])
        self.assertAlmostEqual(self.model.accuracy(a, b), 0.666667, places=5, msg='Not')
        assert isinstance(self.model.accuracy(a,b), float)

    def test_weighted_sum(self):
        batch_size = 7
        inputs = torch.randn((batch_size, self.n_in), dtype=torch.float)
        weights = torch.randn(( self.n_in,self.n_h1), dtype=torch.float)
        biases = torch.randn((self.n_h1), dtype=torch.float)
        outputs = self.model.weighted_sum(inputs, weights, biases)
        assert isinstance(outputs, torch.FloatTensor)
        assert outputs.size() == torch.Size([batch_size, self.n_h1])

    def test_forward(self):
        batch_size = 11
        inputs = torch.randn((batch_size, self.n_in), dtype=torch.float)

        outputs = self.model.forward(inputs)
        assert isinstance(outputs, torch.FloatTensor) 
        assert outputs.size() == torch.Size([batch_size, self.n_out])

    def test_backward(self):
        # settings
        batch_size = 7
        # tensors
        inputs = torch.randn((batch_size, self.n_in), dtype=torch.float)
        # outputs = torch.rand((batch_size, self.n_out), dtype=torch.float)
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)
        one_hot = torch.FloatTensor(labels.size(0), 10).zero_()
        target = one_hot.scatter_(1, labels.view((labels.size(0),1)), 1)            
        y = target
        
        # required to set caches
        outputs = self.model.forward(inputs)

        dw1, db1, dw2, db2, dw3, db3 = self.model.backward(inputs, labels, outputs)
        assert isinstance(dw1, torch.FloatTensor)
        assert dw1.size() == self.model.weights['w1'].size()
        assert isinstance(dw2, torch.FloatTensor)
        assert dw2.size() == self.model.weights['w2'].size()
        assert isinstance(dw3, torch.FloatTensor)
        assert dw3.size() == self.model.weights['w3'].size()
        assert isinstance(db1, torch.FloatTensor)
        assert db1.size() == self.model.biases['b1'].size()
        assert isinstance(db2, torch.FloatTensor)
        assert db2.size() == self.model.biases['b2'].size()
        assert isinstance(db3, torch.FloatTensor)
        assert db3.size() == self.model.biases['b3'].size()

    def test_calculate_grad(self):
        # settings
        batch_size = 7
        # tensors
        inputs = torch.randn((batch_size, self.n_in), dtype=torch.float)
        # outputs = torch.rand((batch_size, self.n_out), dtype=torch.float)
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)
        # required to set caches
        outputs = self.model.forward(inputs)
        self.label=labels.shape[0]
        
        dout = torch.randn((batch_size, self.n_out), dtype=torch.float)
        d2 = torch.randn((batch_size, self.n_h2), dtype=torch.float)
        d1 = torch.randn((batch_size, self.n_h1), dtype=torch.float)

        dw1, db1, dw2, db2, dw3, db3 = self.model.calculate_grad(inputs, d1, d2, dout)
        assert isinstance(dw1, torch.FloatTensor)
        assert dw1.size() == self.model.weights['w1'].size()
        assert isinstance(dw2, torch.FloatTensor)
        assert dw2.size() == self.model.weights['w2'].size()
        assert isinstance(dw3, torch.FloatTensor)
        assert dw3.size() == self.model.weights['w3'].size()
        assert isinstance(db1, torch.FloatTensor)
        assert db1.size() == self.model.biases['b1'].size()
        assert isinstance(db2, torch.FloatTensor)
        assert db2.size() == self.model.biases['b2'].size()
        assert isinstance(db3, torch.FloatTensor)
        assert db3.size() == self.model.biases['b3'].size()

if __name__ == '__main__':
    unittest.main()
