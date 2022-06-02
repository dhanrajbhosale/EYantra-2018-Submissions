

# NOTE: You can only use Tensor API of PyTorch

import torch
import math
# Extra TODO: Document with proper docstring
def sigmoid(z):
    """Calculates sigmoid values for tensors

    """
     
    
    activation = 1/(1 + math.e**(-z))
    
    return activation

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    """
    grad_sigmoid = z*(1-z)
    return grad_sigmoid

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors

    """
    softmax=torch.exp(x)/torch.sum(torch.exp(x),dim=1).view(-1,1)
    
    return softmax# NOTE: You can only use Tensor API of PyTorch


if __name__ == "__main__":
    pass