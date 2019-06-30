####### DEPENDENCIES #######
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import numpy as np
import pandas as pd
import supportFunctions
import supportClasses

############################
USE_GPU = True
dtype = torch.float32

device = supportFunctions.getDevice(USE_GPU)

model_filename = 'resnet_model.pth'

model = torchvision.models.resnet34(pretrained = True) #Edit model to match the loaded model!
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

model, optimizer, epochs = supportFunctions.loadModel(model, filename = model_filename)
model.to(device = device)
model.eval()

isic_data = supportClasses.ISICDataset(csv_file = 'Data/HAM10000_metadata.csv',
                        root_dir = 'Data/Images/')
_, _, test = isic_data.loadDataset()

batch_size = test.batch_size
n = 0    
output = torch.zeros(1512, 7)
softmax = nn.Softmax()
with torch.no_grad():
    #Running model
    for x, y in test:
        x = x.to(device = device, dtype = dtype)
        scores = model(x)
        output[batch_size * n : batch_size * (n+1)] = scores
        n += 1

    #Creation of output CSV-file
    a = torch.std(output)
    #output = supportFunctions.sigmoidConversion(output, a = 1/a, threshold = 10)  
    output = softmax(output)
    imgnames = []
    
    for i in range(0, 1512):
        name = test.dataset.imgs[i]
        name = name[0]
        name = name[len(name) - 16 : len(name) - 4]
        imgnames.append(name)
    output = np.array(output)
    list_of_tuples = list(zip(imgnames, output[:,0], output[:,1], output[:,2],
                              output[:,3], output[:,4], output[:,5], output[:,6]))
    df = pd.DataFrame(list_of_tuples, columns = ['image', 'AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC'])
    df.to_csv(r'test_output.csv')