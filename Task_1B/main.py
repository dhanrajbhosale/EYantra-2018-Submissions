
# Homecoming (eYRC-2018): Task 1B
# Fruit Classification with a CNN
import os
import re
import torch
import csv
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from PIL import Image

from model import FNet
from dataset import ImageDataset
# import required modules

def train_model(dataset_path, debug=False, destination_path='', save=False):
        """Trains model with set hyper-parameters and provide an option to save the model.

        This function should contain necessary logic to load fruits dataset and train a CNN model on it. It should accept dataset_path which will be path to the dataset directory. You should also specify an option to save the trained model with all parameters. If debug option is specified, it'll print loss and accuracy for all iterations. Returns loss and accuracy for both train and validation sets.

        Args:
                dataset_path (str): Path to the dataset folder. For example, '../Data/fruits/'.
                debug (bool, optional): Prints train, validation loss and accuracy for every iteration. Defaults to False.
                destination_path (str, optional): Destination to save the model file. Defaults to ''.
                save (bool, optional): Saves model if True. Defaults to False.

        Returns:
                loss (torch.tensor): Train loss and validation loss.
                accuracy (torch.tensor): Train accuracy and validation accuracy.
        """
        # Write your code here
        # loading dataframes using dataset module 
        # loading dataframes using dataset module 
        data_transforms = {'train': transforms.Compose([transforms.ToTensor()]),'val': transforms.Compose([transforms.ToTensor()]),}
        image_datasets = {'train': dataset.ImageDataset(df_train, transform=data_transforms['train']),'val': dataset.ImageDataset(df_test, transform=data_transforms['val'])}
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=4,shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=4,shuffle=True)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Hyper parameters
        num_epochs = 5
        num_classes = 10
        batch_size = 100
        learning_rate = 0.001
        net = model.FNet(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        total_step = len(train_loader)
        for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        # Forward pass
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                        if (i+1) % 100 == 0:
                                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Test the model
        net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
                correct = 0
                total = 0
                cnt=0
                classes = ['Apple', 'Banana','Orange','Pineapple','Strawberry']
                for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                        tmp = []
                        for i in labels:
                                tmp.append(i.item())
                        cntr = 0
                        for i in predicted:
                                print("actual: ",classes[i.item()],"\t Predicted: ",classes[tmp[cntr]])
                                if i.item() == tmp[cntr]:
                                        cnt+=1
                                cntr+=1
                        total+=4
        
    

        print('Test Accuracy of the model on the test images: {} %'.format(100 * cnt / total))        # Save the model checkpoint
        #torch.save(model.state_dict(), 'model.ckpt')

      



        # NOTE: Make sure you use torch.device() to use GPU if available
        pass

if __name__ == "__main__":
        train_model('../Data/fruits/', save=True, destination_path='./')
