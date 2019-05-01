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
        self.training_size = 1000
        self.test_size = 200
        self.resize = True
        self.image_size = [120, 90]
        print('ISICDataset Class Successfully Initiated \n')
    
    def getTransfrom(self):
        t = T.Compose([T.Resize(self.image_size), T.ToTensor()])
        return t
    
    def loadDataset(self):
        if self.resize:
            transform = self.getTransfrom()
        else:
            transform = T.ToTensor()
            
        train_dataset = dset.ImageFolder(root = os.path.join(self.root_dir + 'Train/'),
                                         transform = transform)
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size = 100,
                                  #sampler = sampler.SubsetRandomSampler(range(self.training_size)),
                                  shuffle = True)
        
        val_dataset = dset.ImageFolder(root = os.path.join(self.root_dir + 'Validation/'),
                                       transform = transform)
        val_loader = DataLoader(dataset = val_dataset,
                                batch_size = 100,
                                #sampler = sampler.SubsetRandomSampler(range(self.test_size)),
                                shuffle = True)
        
        test_dataset = dset.ImageFolder(root = os.path.join(self.root_dir + 'Test/'),
                                        transform = transform)
        test_loader = DataLoader(dataset = test_dataset,
                                 batch_size = 100,
                                 #sampler = sampler.SubsetRandomSampler(range(self.test_size + 1,2*self.test_size)),
                                 shuffle = True)
        
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
    