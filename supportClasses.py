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
        self.normalize = False #Edit this variable in function loadDataset()
        self.image_size = [224, 224] #original [600x450]
        self.mu = [0.779659, 0.54361504, 0.56447685]
        self.sigma = [0.08348679, 0.11146858, 0.12505478]
        print('ISICDataset Class Successfully Initiated')
    
    def getTransfrom(self, mu = [], sigma = []):
        """
        Creates the transform tuple
        """
        if self.normalize:
            t = T.Compose([T.Resize(self.image_size), T.ToTensor(),
                           T.Normalize((mu[0], mu[1], mu[2]),(sigma[0], sigma[1], sigma[2]))])
        else:
            t = T.Compose([T.Resize(self.image_size), T.ToTensor()])
        return t
    
    def getStatistics(self, data):
        """
        Calculates statistics of the dataset (mean and standard deviation per channel)
        """
        muR = []
        muG = []
        muB = []
        stdR = []
        stdG = []
        stdB = []
        
        for n in range(len(data)):
            print(100 * (n/len(data)), '%')
            for channel in range(0, 3):
                mu = data[n][0][channel].mean()
                sigma = data[n][0][channel].std()
                if channel == 0:
                    muR.append(mu)
                    stdR.append(sigma)
                elif channel == 1:
                    muG.append(mu)
                    stdG.append(sigma)
                elif channel == 2:
                    muB.append(mu)
                    stdB.append(sigma)
                    
        means = [np.mean(muR), np.mean(muG), np.mean(muB)]
        stds = [np.mean(stdR), np.mean(stdG), np.mean(stdB)]
        return means, stds
    
    def loadDataset(self):
        """
        Loads the dataset
        """
        if self.resize:
            transform = self.getTransfrom()
        else:
            transform = T.ToTensor()
            
        train_dataset = dset.ImageFolder(root = os.path.join(self.root_dir + 'Train/'),
                                         transform = transform)
        
        self.normalize = False #Edit variable here
        if self.normalize:
            #m, s = self.getStatistics(train_dataset) #Uncomment to calculate stats; takes a long time!
            m = self.mu
            s = self.sigma
            if self.resize and self.normalize:
                transform = self.getTransfrom(mu = m, sigma = s)
            else:
                print('[Warning] To use normalization, resizing must be enabled. Normalization will not be not used for this run.')
                
        train_dataset = dset.ImageFolder(root = os.path.join(self.root_dir + 'Train/'),
                                 transform = transform)
        
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size = 20,
                                  #sampler = sampler.SubsetRandomSampler(range(self.training_size)),
                                  shuffle = True)
        
        val_dataset = dset.ImageFolder(root = os.path.join(self.root_dir + 'Validation/'),
                                       transform = transform)
        val_loader = DataLoader(dataset = val_dataset,
                                batch_size = 20,
                                #sampler = sampler.SubsetRandomSampler(range(self.test_size)),
                                shuffle = True)
        
        test_dataset = dset.ImageFolder(root = os.path.join(self.root_dir + 'Test/'),
                                        transform = transform)
        test_loader = DataLoader(dataset = test_dataset,
                                 batch_size = 54,
                                 #sampler = sampler.SubsetRandomSampler(range(self.test_size + 1,2*self.test_size)),
                                 shuffle = False)
        
        print('Dataset Successfully Loaded')
        return train_loader, val_loader, test_loader

    
    def plotRandomSample(self):
        """
        Plots random image from the database with the corresponding diagnosis
        CURRENTLY UNUSED AND NOT FUNCTIONING
        """
        sample = int(10000 * np.random.rand())
        img_name = self.database.image_id[sample]
        dir_name = self.root_dir + '/Images/' + img_name + '.jpg'
        plt.imshow(io.imread(os.path.join(dir_name)))
        print('Random Sample from Dataset')
        plt.show()
        print('Diagnosis:', self.database.dx[sample], '\n')
        return
    