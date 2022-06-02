# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:01:04 2018

@author: Attarde
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import os
import argparse
from pathlib import Path
plt.ion() 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Dictionary of Animals
animal_classes=['arctic fox','bear','bee','butterfly','cat','cougar','cow','coyote','crab','crocodile','deer','dog','eagle','elephant','fish','frog','giraffe','goat','hippo','horse','kangaroo','lion','monkey','otter','panda','parrot','penguin','raccoon','rat','seal','shark','sheep','skunk','snake','snow leopard','tiger','yak','zebra']
#Dictionary of Habitats
habitat_classes = ['baseball','basketball court','beach','circular farm','cloud','commercial area','dense residential','desert','forest','golf course','harbor','island','lake','meadow','medium residential area','mountain','rectangular farm','river','sea glacier','shrubs','snowberg','sparse residential area','thermal power station','wetland']

def argumentparse():                                # argparse commands for api
    ap = argparse.ArgumentParser(conflict_handler='resolve')
    # path to the animal image. In command prompt type python main.py -a path
    ap.add_argument("-a", "--animal", required = True, help = "Path to the animal image")
    # path to the habitat image. In command prompt type python main.py -h path
    ap.add_argument("-h", "--habitat", required = True, help="Path to the habitat image")
    # path to the animal model. In command prompt type python main.py -amod animals.pth
    ap.add_argument("-amod", "--animal_model", help="path to animal_model",default='animals.pth')
    # path to the habitat model. In command prompt type python main.py -hmod habitats.pth 
    ap.add_argument("-hmod","--habitat_model",help="path to habitat_model",default='habitats.pth')
    args = vars(ap.parse_args())
    return args

# transformations on image
trans = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    transforms.Resize(360),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args=argumentparse()
animal_image=Image.open(Path(args["animal"])) # load animal image
habitat_image=Image.open(Path(args["habitat"]))# load habitat image

animal_input=trans(animal_image) # Apply transform on animal image
animal_input = animal_input.view(1, 3, 299,299) # Making it suitable as per input to the animal  model
animal_input=animal_input.to(device)

habitat_input=trans(habitat_image) # Apply transform on habitat image
habitat_input = habitat_input.view(1, 3, 299,299) # Making it suitable as per input to the habitat  model
habitat_input=habitat_input.to(device)

model_animal = torchvision.models.inception_v3() # animal model class
num_ftrs = model_animal.fc.in_features 
model_animal.fc = nn.Linear(num_ftrs, 38)  # making the last layer output as per our number of classes
model_animal.aux_logits = False
model_animal.load_state_dict(torch.load(args["animal_model"], map_location='cpu')) # loading animal model
model_animal.eval() # setting the model to eval mode
output=model_animal(animal_input)
#print(output)
animal_name=torch.max(output,1)[1]# predicting the animal name

model_habitat = torchvision.models.inception_v3() # habitat model class
num_ftrs = model_habitat.fc.in_features
model_habitat.fc = nn.Linear(num_ftrs, 24)  # making the last layer output as per our number of classes
model_habitat.aux_logits = False
model_habitat.load_state_dict(torch.load(args["habitat_model"], map_location='cpu')) # loading habitat model
model_habitat.eval() # setting the model to eval mode
output1=model_habitat(habitat_input)
#print(output)
habitat_name=torch.max(output1,1)[1] # predicting the habitat name
print("Animal name is:")
print(animal_classes[animal_name])
print("Habitat name is:")
print(habitat_classes[habitat_name])

