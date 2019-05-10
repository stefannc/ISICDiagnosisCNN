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

model_filename = 'checkpoint.pth'

model = torchvision.models.resnet101(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

model, optimizer, epochs = supportFunctions.loadModel(model, filename = model_filename)
model.to(device = device)
model.eval()

isic_data = supportClasses.ISICDataset(csv_file = 'Data/HAM10000_metadata.csv',
                        root_dir = 'Data/Images/')
train, val, test = isic_data.loadDataset()

print('Do you want to create an output .csv file? [y/n]')
ans = 'y'# input()
if ans == 'y':
    CREATE_CSV = True
else:
    CREATE_CSV = False
    
n = 0    
output = torch.zeros(1000, 7)
confusion_matrix = torch.zeros(7, 7)
with torch.no_grad():
    for x, y in val:
        x = x.to(device = device, dtype = dtype)
        y = y.to(device = device, dtype = torch.long)
        scores = model(x)
        output[100 * n : 100 * (n+1)] = scores
        n += 1
        _, preds = scores.max(1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    if CREATE_CSV:
        a = torch.std(output)
        output = supportFunctions.sigmoidConversion(output, a = 1/a, threshold = 0.5)    
        imgnames = []
        for i in range(0, 1000):
            name = val.dataset.imgs[i]
            name = name[0]
            name = name[len(name) - 16 : len(name)]
            imgnames.append(name)
        output = np.array(output)
        list_of_tuples = list(zip(imgnames, output[:,0], output[:,1], output[:,2],
                                  output[:,3], output[:,4], output[:,5], output[:,6]))
        df = pd.DataFrame(list_of_tuples, columns = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
        df.to_csv(r'output.csv')
            
    print(confusion_matrix)
    TP = confusion_matrix.diag()
    TN = []
    FP = []
    FN = []
    for c in range(7):
        idx = torch.ones(7).byte()
        idx[c] = 0
        TN.append(confusion_matrix[idx.nonzero()[:,None], idx.nonzero()].sum())
        FP.append(confusion_matrix[idx, c].sum())
        FN.append(confusion_matrix[c, idx].sum())
    
    printing = True
    tboard = False
    supportFunctions.performance(TP, TN, FP, FN, printing, tboard)

    
    
    
    
    