import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



# Convolutional neural network (two convolutional layers)
class FNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(25*25*32, num_classes)
        pass
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out






if __name__ == "__main__":
    net = FNet()
