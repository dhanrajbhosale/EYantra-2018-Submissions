

# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    
    """Calculates cross entropy loss given outputs and actual labels

    """  

    a=outputs
    elements=a.numel()
    rows=len(a)
    columns=elements/rows
    y=int(columns)

    a=outputs
    one_hot = torch.FloatTensor(labels.size(0),y).zero_()
    target = one_hot.scatter_(1, labels.view((labels.size(0),1)), 1)            
    y = target
    z=torch.sum(-y*torch.log(a)-(1-y)*torch.log(1-a))
    #z=float(z)
    #print("loss",z)
    return z.item()
    #return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """

    #one_hot = torch.FloatTensor(labels.size(0), 10).zero_()
    #target = one_hot.scatter_(1, labels.view((labels.size(0),1)), 1)            
    #y = target
    t=outputs-labels
    return t
    #return avg_grads




if __name__ == "__main__":
    pass
