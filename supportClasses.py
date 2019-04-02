import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

##############################################################################
"""
Flatten 
"""

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
##############################################################################   
"""
ISICDataset
"""

class ISICDataset():
    
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to csv file
            root_dir (string): Image directory
        """
        self.database = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.size = self.database.shape[0]
        self.training_size = 128
        self.test_size = 64
        print('ISICDataset Class Successfully Initiated \n')
    
    def loadDataset(self):
        train_dataset = dset.ImageFolder(root = os.path.join(self.root_dir),
                                         transform = T.ToTensor())
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size = 64,
                                  sampler = sampler.SubsetRandomSampler(range(self.training_size)))
        
        val_dataset = dset.ImageFolder(root = os.path.join(self.root_dir),
                                       transform = T.ToTensor())
        val_loader = DataLoader(dataset = val_dataset,
                                batch_size = 64,
                                sampler = sampler.SubsetRandomSampler(range(self.training_size, 2 * self.training_size)))
        
        test_dataset = dset.ImageFolder(root = os.path.join(self.root_dir),
                                        transform = T.ToTensor())
        test_loader = DataLoader(dataset = test_dataset,
                                 batch_size = 64,
                                 sampler = sampler.SubsetRandomSampler(range(2 * self.training_size, 3 * self.training_size)))
        
        print('Dataset Successfully Loaded \n')
        return train_loader, val_loader, test_loader

    
    def plotRandomSample(self):
        #Plots random image from the database with the corresponding diagnosis
        sample = int(10000 * np.random.rand())
        img_name = self.database.image_id[sample]
        dir_name = self.root_dir + '/Images/' + img_name + '.jpg'
        plt.imshow(io.imread(os.path.join(dir_name)))
        print('Random Sample from Dataset')
        plt.show()
        print('Diagnosis:', self.database.dx[sample], '\n')
        return
    