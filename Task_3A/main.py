from __future__ import print_function, division
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

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
#import copy
from PIL import Image

plt.ion() 
loc=21
char="A"
num=6
# Dictionary of animals
animal_classes=['arctic fox','bear','bee','butterfly','cat','cougar','cow','coyote','crab','crocodile','deer','dog','eagle','elephant','fish','frog','giraffe','goat','hippo','horse','kangaroo','lion','monkey','otter','panda','parrot','penguin','raccoon','rat','seal','shark','sheep','skunk','snake','snow leopard','tiger','yak','zebra']
# Dictionary of Habitats
habitat_classes = ['baseball','basketball court','beach','circular farm','cloud','commercial area','dense residential','desert','forest','golf course','harbor','island','lake','meadow','medium residential area','mountain','rectangular farm','river','sea glacier','shrubs','snowberg','sparse residential area','thermal power station','wetland']
# Finding habitat locations in the image on the basis of found centroid of habitat contour and comparing with manual simulations
def habitatlocation(scale,cx,cy):
    if ((cx>=390*scale and cx<=690*scale) and (cy>=390*scale and cy<=690*scale)):
        location=loc
    if ((cx>=390*scale and cx<=690*scale) and (cy>=700*scale and cy<=1000*scale)):
        location=loc-5
    if ((cx>=390*scale and cx<=690*scale) and (cy>=1010*scale and cy<=1310*scale)):
        location=loc-10
    if ((cx>=390*scale and cx<=690*scale) and (cy>=1320*scale and cy<=1620*scale)):
        location=loc-15
    if ((cx>=390*scale and cx<=690*scale) and (cy>=1630*scale and cy<=1930*scale)):
        location=loc-20
        
    if ((cx>=700*scale and cx<=1000*scale) and (cy>=390*scale and cy<=690*scale)):
        location=loc+1
    if ((cx>=700*scale and cx<=1000*scale) and (cy>=700*scale and cy<=1000*scale)):
        location=loc-4
    if ((cx>=700*scale and cx<=1000*scale) and (cy>=1010*scale and cy<=1310*scale)):
        location=loc-9
    if ((cx>=700*scale and cx<=1000*scale) and (cy>=1320*scale and cy<=1620*scale)):
        location=loc-14
    if ((cx>=700*scale and cx<=1000*scale) and (cy>=1630 and cy<=1930*scale)):
        location=loc-19
        
    if ((cx>=1010*scale and cx<=1310*scale) and (cy>=390*scale and cy<=690*scale)):
        location=loc+2
    if ((cx>=1010*scale and cx<=1310*scale) and (cy>=700*scale and cy<=1000*scale)):
        location=loc-3
    if ((cx>=1010*scale and cx<=1310*scale) and (cy>=1010*scale and cy<=1310*scale)):
        location=loc-8
    if ((cx>=1010*scale and cx<=1310*scale) and (cy>=1320*scale and cy<=1620*scale)):
        location=loc-13
    if ((cx>=1010*scale and cx<=1310*scale) and (cy>=1630*scale and cy<=1930*scale)):
        location=loc-18
    
    if ((cx>=1320*scale and cx<=1620*scale) and (cy>=390*scale and cy<=690*scale)):
        location=loc+3
    if ((cx>=1320*scale and cx<=1620*scale) and (cy>=700*scale and cy<=1000*scale)):
        location=loc-2
    if ((cx>=1320*scale and cx<=1620*scale) and (cy>=1010*scale and cy<=1310*scale)):
        location=loc-7
    if ((cx>=1320*scale and cx<=1620*scale) and (cy>=1320*scale and cy<=1620*scale)):
        location=loc-12
    if ((cx>=1320*scale and cx<=1620*scale) and (cy>=1630*scale and cy<=1930*scale)):
        location=loc-17
        
    if ((cx>=1630*scale and cx<=1930*scale) and (cy>=390*scale and cy<=690*scale)):
        location=loc+4
    if ((cx>=1630*scale and cx<=1930*scale) and (cy>=700*scale and cy<=1000*scale)):
        location=loc-1
    if ((cx>=1630*scale and cx<=1930*scale) and (cy>=1010*scale and cy<=1310*scale)):
        location=loc-6
    if ((cx>=1630*scale and cx<=1930*scale) and (cy>=1320*scale and cy<=1620*scale)):
        location=loc-11
    if ((cx>=1630*scale and cx<=1930*scale) and (cy>=1630*scale and cy<=1930*scale)):
        location=loc-16
    
    return location

# Finding animal locations in the image on the basis of found centroid of animal contour and comparing with manual simulations
def animallocation(scale,cx,cy):    
    if ((cx>=130*scale and cx<=260*scale) and (cy>=130*scale and cy<=260*scale)):
        ch = chr(ord(char) + 0)
        nm=num-0
        location=ch+str(nm)
    if ((cx>=630*scale and cx<=760*scale) and (cy>=130*scale and cy<=260*scale)):
        ch = chr(ord(char) + 1)
        nm=num-0
        location=ch+str(nm)
    if ((cx>=940*scale and cx<=1070*scale) and (cy>=130*scale and cy<=260*scale)):
        ch = chr(ord(char) + 2)
        nm=num-0
        location=ch+str(nm)
    if ((cx>=1250*scale and cx<=1380*scale) and (cy>=130*scale and cy<=260*scale)):
        ch= chr(ord(char) + 3)
        nm=num-0
        location=ch+str(nm)
    if ((cx>=1560*scale and cx<=1690*scale) and (cy>=130*scale and cy<=260*scale)):
        ch = chr(ord(char) + 4)
        nm=num-0
        location=ch+str(nm)
    if ((cx>=2060*scale and cx<=2190*scale) and (cy>=130*scale and cy<=260*scale)):
        ch = chr(ord(char) + 5)
        nm=num-0
        location=ch+str(nm)
        
        
    if ((cx>=130*scale and cx<=260*scale) and (cy>=630*scale and cy<=760*scale)):
        ch = chr(ord(char) + 0)
        nm=num-1
        location=ch+str(nm)
    if ((cx>=130*scale and cx<=260*scale) and (cy>=940*scale and cy<=1070*scale)):
        ch = chr(ord(char) + 0)
        nm=num-2
        location=ch+str(nm)
    if ((cx>=130*scale and cx<=260*scale) and (cy>=1250*scale and cy<=1380*scale)):
        ch = chr(ord(char) + 0)
        nm=num-3
        location=ch+str(nm)
    if ((cx>=130*scale and cx<=260*scale) and (cy>=1560*scale and cy<=1690*scale)):
        ch = chr(ord(char) + 0)
        nm=num-4
        location=ch+str(nm)
    
    if ((cx>=130*scale and cx<=260*scale) and (cy>=2060*scale and cy<=2190*scale)):
        ch = chr(ord(char) + 0)
        nm=num-5
        location=ch+str(nm)
    if ((cx>=630*scale and cx<=760*scale) and (cy>=2060*scale and cy<=2190*scale)):
        ch = chr(ord(char) + 1)
        nm=num-5
        location=ch+str(nm)
    if ((cx>=940*scale and cx<=1070*scale) and (cy>=2060*scale and cy<=2190*scale)):
        ch = chr(ord(char) + 2)
        nm=num-5
        location=ch+str(nm)
    if ((cx>=1250*scale and cx<=1380*scale) and (cy>=2060*scale and cy<=2190*scale)):
        ch = chr(ord(char) + 3)
        nm=num-5
        location=ch+str(nm)
    if ((cx>=1560*scale and cx<=1690*scale) and (cy>=2060*scale and cy<=2190*scale)):
        ch = chr(ord(char) + 4)
        nm=num-5
        location=ch+str(nm)
    if ((cx>=2060*scale and cx<=2190*scale) and (cy>=2060*scale and cy<=2190*scale)):
        ch = chr(ord(char) + 5)
        nm=num-5
        location=ch+str(nm)

        
    if ((cx>=2060*scale and cx<=2190*scale) and (cy>=630*scale and cy<=760*scale)):
        ch = chr(ord(char) + 5)
        nm=num-1
        location=ch+str(nm)
    if ((cx>=2060*scale and cx<=2190*scale) and (cy>=940*scale and cy<=1070*scale)):
        ch = chr(ord(char) + 5)
        nm=num-2
        location=ch+str(nm)
    if ((cx>=2060*scale and cx<=2190*scale) and (cy>=1250*scale and cy<=1380*scale)):
        ch = chr(ord(char) + 5)
        nm=num-3
        location=ch+str(nm)
    if ((cx>=2060*scale and cx<=2190*scale) and (cy>=1560*scale and cy<=1690*scale)):
        ch = chr(ord(char) + 5)
        nm=num-4
        location=ch+str(nm)
        
    return location

def centroid(cnt):
    M=cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])                                    # Finding Centroid
    c=img[cy,cx]
    return c,cx,cy

def drawboundary(cnt,scale):
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),int(3*scale))              # Drawing boundary for contour
    return x,y,w,h

def binaryimage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           # Converting image to grayscale
    #cv2.imshow('gray', gray)
    ret,thresh = cv2.threshold(gray,254,255,1)             # Converting grayscale image to threshold
    #cv2.imshow('thresh', thresh)
    return ret,thresh

def cutcontours(flg,location):                              # Cutting contours and saving by name of their locations
    if flg==0:
    
        new_img=img[y+6:y+h-6,x+6:x+w-6]                    # Habitat locations
       # path= 'C:/Users/Attarde/Desktop/Task 3/Val Habitats/Val'
        cv2.imwrite((str(location)+".png"),new_img)
    else:
        new_img=img[y+10:y+h-10,x+10:x+w-10]                    # Animal locations
        #new_img=get_square(new_img,(299,299),cv2.INTER_CUBIC)
       # path= 'C:/Users/Attarde/Desktop/Task 3/Val Animals'
        cv2.imwrite((str(location)+".png"),new_img)
    


 
         
def argumentparse():                                # argparse commands for api
    ap = argparse.ArgumentParser()
    # Image Path(-i,--image) is specified. In command prompt type: python main.py -i img_name.png
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    # Save(-s,--save) command. In command prompt type: python main.py -i img_name.png -s save_image.png
    ap.add_argument("-s", "--save",action="store", help="path to output image")
    # path to the animal model. In command prompt type python main.py -amod animals.pth
    ap.add_argument("-amod", "--animal_model", help="path to animal_model",default='animals.pth')
    # path to the habitat model. In command prompt type python main.py -hmod habitats.pth 
    ap.add_argument("-hmod","--habitat_model",help="path to habitat_model",default='habitats.pth')

    args = vars(ap.parse_args())
    return args

args=argumentparse()
img = cv2.imread(args["image"])    # Reading image
#print('Original Dimensions : ',img.shape)
originalshape=img.shape[0]       # Saving original dimensions of image
scale=originalshape/2319
 
ret,thresh=binaryimage(img)                 # thresh image
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))       # kernel which decides the nature of operation for morphological transformation

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Opening is just another name of erosion followed by dilation. It is useful in removing noise.
#cv2.imshow('opening', opening)
kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))                                  
dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel1)  #  Morphologically dilating the opening image.
#cv2.imshow('dilation', dilate)

# Finding contours in the image and assigning all contours to same hierarchy level using cv2.RETR_LIST
im2, contours, hierarchy = cv2.findContours(dilate,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   
locations_habitats=[]
habitats=[]
animals=[]
for cnt in contours:                                    # looping through the found contours

    area=cv2.contourArea(cnt)                                # finding area of individual contour
   # print(area)
    if (area>(75000*(scale*scale))):                                 # Contour whose area is greater than specified area for habitat location
        flg=0
        flag=0
        c,cx,cy=centroid(cnt)                               # finding centroid of individual contour
        #print(cx,cy)
        #cv2.circle(img,(cx,cy), 3, (0,0,255), -1)
        #print(cx,cy)
        location= habitatlocation(scale,cx,cy)                       # finding habitat location in the image
        locations_habitats.append(location) # storing habitats location
        x,y,w,h=drawboundary(cnt,scale)                            # drawing boundary to contour
        cutcontours(flg,location) #  cut and save habitat contours
        cv2.putText(img,str(location), (cx,cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, (2.30*scale), (255,255,255))
#cv2.imshow('First',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

ret,thresh=binaryimage(img)                                  # thresh image
edged = cv2.Canny(thresh,254,255,apertureSize = 7)          #  finding edges in the image through canny edge detection
#cv2.imshow('opening', edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Finding only external contours meaning all parent contours using cv2.RETR_EXTERNAL

im3, contours1, hierarchy1 = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
locations_animals=[]
for cnt in contours1:                               # looping through the found contours
    area=cv2.contourArea(cnt)                       # finding area of individual contour
    #print(area)
    if(area>=(16000*(scale*scale)) and area<=(24000*(scale*scale))):          # Contour whose area is in specified range for animal location   
        flg=1
        c,cx,cy=centroid(cnt)                      # finding centroid of individual contour
        
# Different locations around the centroid of individual contour to detect presence of another contour
        c1=img[cy,cx-int(10*scale)]
        c2=img[cy,cx+int(10*scale)]
        c3=img[cy-int(10*scale),cx]
        c4=img[cy+int(10*scale),cx]
#        cv2.circle(img,(cx,cy-10), 3, (0,0,255), -1)
#        cv2.circle(img,(cx,cy+10), 3, (0,0,255), -1)
#        cv2.circle(img,(cx-10,cy), 3, (0,0,255), -1)
#        cv2.circle(img,(cx+10,cy), 3, (0,0,255), -1)
#        cv2.circle(img,(cx,cy), 3, (0,0,255), -1)
        
        # Detecting presence of animal in animal locations by detecting presence of another contour
        if (c[0]!=255 or c[1]!=255 or c[2]!=255) or (c1[0]!=255 or c1[1]!=255 or c1[2]!=255) or (c2[0]!=255 or c2[1]!=255 or c2[2]!=255) or (c3[0]!=255 or c3[1]!=255 or c3[2]!=255) or (c4[0]!=255 or c4[1]!=255 or c4[2]!=255) :
            location=animallocation(scale,cx,cy)  # finding animal location in the image
            locations_animals.append(location) # storing animals locations
            x,y,w,h=drawboundary(cnt,scale)    # drawing boundary to contour
            cutcontours(flg,location)      #  cut and save animal contours
            cv2.putText(img,str(location), (cx,cy-int(110*scale)), cv2.FONT_HERSHEY_COMPLEX_SMALL, (2.30*scale), 0)


if args['save']:                # if save command then save output image with original image dimensions
    cv2.imwrite(args["save"], img)  # Saving image
    
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Transformations to be applied on image
trans = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.ToPILImage(),
    transforms.Resize(360),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
 
    
model_habitat = torchvision.models.inception_v3() # habitat model class
num_ftrs = model_habitat.fc.in_features
model_habitat.fc = nn.Linear(num_ftrs, 24)  # making the last layer output as per our number of classes
model_habitat.aux_logits = False
model_habitat.load_state_dict(torch.load(args["habitat_model"], map_location='cpu')) # loading habitat model
model_habitat.eval() # setting the model to eval mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Predicting the names of the habitats from the arena
habitats_1=[] 
for i in range(len(locations_habitats)):
  #  Directory="C:\\Users\Attarde\Desktop\Task 3\Val Habitats\\Val"
    path=(str(locations_habitats[i])+".png")

    image = Image.open(Path(path))
    
    input=trans(image) # Applying transformations to image
    input = input.view(1, 3, 299,299) # Making it suitable as per input to the animal  model
    input=input.to(device)
    model_habitat.eval()
    output=model_habitat(input)
    #print(output)
    pred=torch.max(output,1)[1] # Predicting habitat name
    habitats_1.append(habitat_classes[pred]) # Storing habitat names
    
model_animal = torchvision.models.inception_v3()
num_ftrs = model_animal.fc.in_features
model_animal.fc = nn.Linear(num_ftrs, 38)
#model.fc = nn.Linear(2048, 24)
model_animal.aux_logits = False
model_animal.load_state_dict(torch.load('animals.pth', map_location='cpu'))
model_animal.eval()

# Predicting the names of the habitats from the arena
animals_1=[]
for i in range(len(locations_animals)):
    #Directory="C:\\Users\Attarde\Desktop\Task 3\Val Animals"
    path=(str(locations_animals[i])+".png")
#
    image = Image.open(Path(path))
    input=trans(image) # Applying transformations to image
    input = input.view(1, 3, 299,299) # Making it suitable as per input to the animal  model
    input=input.to(device)
    model_animal.eval()
    output=model_animal(input)
    #print(output)
    pred=torch.max(output,1)[1] # predicting animal name
    animals_1.append(animal_classes[pred]) # Storing animal names

flag=0
final_habitats=[]
for i in range(len(locations_habitats)):
    final_habitats.append(locations_habitats[i])
    #final_habitats.append(str(':'))
    final_habitats.append(habitats_1[i])
    
final_animals=[]
for i in range(len(locations_animals)):
    final_animals.append(locations_animals[i])
    #final_animals.append(str(':'))
    final_animals.append(animals_1[i])
        
print(final_habitats) # Printing the habitats along with their locations
print(final_animals) # Printing the animals along with their locations
cv2.imshow('Output',img)     # Output image
cv2.waitKey(0)
cv2.destroyAllWindows()
