import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage import io

####### GPU SETTINGS #######
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using device:', device)
############################

####### ClASS DEFINITIONS #######

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

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
    
############################

####### FUNCTION DEFINITIONS #######

def trainModel(model, optimizer, epochs=1):
    
    model = model.to(device = device)
    
    for e in range(0, epochs):
        for t, (x, y) in enumerate(train):
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def checkAccuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc
    
############################


####### MAIN CODE #######

isic_data = ISICDataset(csv_file = 'Data/HAM10000_metadata.csv',
                        root_dir = 'Data/')
    
    
isic_data.plotRandomSample()
train, val, test = isic_data.loadDataset()

learning_rate = 1e-2

model = nn.Sequential(
        nn.Conv2d(3, 8, 5, padding = 2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8* 450 * 600, 8)
)

optimizer = optim.SGD(model.parameters(), lr = learning_rate,
                      momentum = 0.9, nesterov = True)

#trainModel(model, optimizer)
#checkAccuracy(test, model)

############################