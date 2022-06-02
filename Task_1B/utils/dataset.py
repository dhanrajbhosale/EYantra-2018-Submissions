# import required libraries
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


def create_meta_csv(dataset_path, destination_path):
    
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """

    # Change dataset path accordingly
    dataset_path = "C:\\Users\\Attarde\\Desktop\\Eyantra 2018\\Task 1\\Task 1B\\Data"
    DATASET_PATH = os.path.abspath(dataset_path)

    if not os.path.exists(os.path.join(DATASET_PATH, "/dataset_attr.csv")):

        # Make a csv with full file path and labels
        with open('C:\\Users\\Attarde\\Desktop\\Eyantra 2018\\Task 1\\Task 1B\\Data\\dataset_attr.csv','w',newline="") as fp:
            SEPARATOR = ","
            a = csv.writer(fp,delimiter=SEPARATOR)
            data = [['Fruit_path', 'Label']]
            a.writerows(data)
            for dirname, dirnames, filenames in os.walk('C:\\Users\\Attarde\\Desktop\\Eyantra 2018\\Task 1\\Task 1B\\Data\\fruits'):
                for subdirname in dirnames:
                    subject_path = os.path.join(dirname, subdirname)
                    for filename in os.listdir(subject_path):
                        abs_path = "%s\%s" % (subject_path, filename)
                        if(re.search("Apple", abs_path)):
                            label = "Apple"
                        elif(re.search("Banana", abs_path)):
                            label = "Banana"
                        elif(re.search("Orange", abs_path)):
                            label = "Orange"
                        elif(re.search("Pineapple", abs_path)):
                            label = "Pineapple"
                        else:
                            label = "Strawberry"

                        data = [[abs_path, label]]
                        a.writerows(data)

            # change destination_path to DATASET_PATH if destination_path is None 
            if destination_path == None:
                destination_path = dataset_path

            # write out as dataset_attr.csv in destination_path directory

            # if no error
            return True

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a 
    fraction in split parameter.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    if create_meta_csv(dataset_path, destination_path=destination_path):
        dframe = pd.read_csv(os.path.join("C:\\Users\\Attarde\\Desktop\\Eyantra 2018\\Task 1\\Task 1B\\Data", 'dataset_attr.csv'))

    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        
        dframe = shuffle(dframe)
        pass 

    if split != None:
        train_set, test_set = train_test_split(dframe, split)
        return dframe, train_set, test_set 
    
    return dframe

def train_test_split(dframe, split_ratio):
    """Splits the dataframe into train and test subset dataframes.

    Args:
        split_ration (float): Divides dframe into two splits.

    Returns:
        train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
        test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
    """
    # divide into train and test dataframes
    train_data  = dframe.iloc[0:int(len(dframe)*split_ratio),:]
    test_data = dframe.iloc[len(dframe)-(len(dframe)-len(train_data)):,:]
    return train_data, test_data



class ImageDataset(Dataset):
    """Image Dataset that works with images
    
    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.
    
    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """
    
    def __init__(self, data,transform):
    
        self.data = data
        self.transform = transform
        self.classes = ['Apple', 'Banana','Orange','Pineapple','Strawberry']
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path=self.data.iloc[idx]['Fruit_path']
        image = Image.open(img_path)
        
        label = self.classes.index(self.data.iloc[idx]['Label'])
        
        
        if self.transform:
            image = self.transform(image)

        return (image, label)


if __name__ == "__main__":
    # test config
    dataset_path = '../Data/fruits'
    dest = '../Data/fruits'
    classes = 5
    total_rows = 4323
    randomize = True
    clear = True
    
    # test_create_meta_csv()
    df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path, destination_path=dest, randomize=randomize, split=0.99)
    expor_csv = trn_df.to_csv(r'C:\\Users\\Attarde\\Desktop\\Eyantra 2018\\Task 1\\Task 1B\\Data\\traineddata.csv',header=True,index=True)
    export_csv = tst_df.to_csv(r'C:\Users\\Attarde\\Desktop\\Eyantra 2018\\Task 1\\Task 1B\\Data\\testdata.csv',header=True,index=True)


    print(df.describe())
    print(trn_df.describe())
    print(tst_df.describe())

