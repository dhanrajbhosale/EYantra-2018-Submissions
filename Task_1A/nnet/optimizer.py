

# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr):
    """Mini-batch gradient descent
    """
    weights['w1'] -= lr*dw1 
    weights['w2'] -= lr*dw2 
    weights['w3'] -= lr*dw3 
    biases['b1'] -= lr*db1 
    biases['b2'] -= lr*db2 
    biases['b3'] -= lr*db3 

    return weights, biases



if __name__ == "__main__":
    pass