import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import supportFunctions
import supportClasses

####### FUNCTION DEFINITIONS #######

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.normal(m.weight)

def trainModel(model, optimizer, data, epochs=1):
    
    model = model.to(device = device)
    traindata = data
    lossWeights = [1/262, 1/411, 1/880, 1/92, 1/890, 1/5364, 1/114]
    lossWeights = torch.tensor(np.multiply(5364, lossWeights), dtype = dtype).cuda()
    
    for e in range(0, epochs):
        for t, (x, y) in enumerate(traindata):
            print(type(x))
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            
            scores = model(x)
            #loss = F.cross_entropy(scores, y, weight = lossWeights)
            loss = nn.BCELoss()
            output = loss(scores, y)
            print('Loss:', output.item())
            
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            if (t % 10 == 0 and t != 0):
                checkAccuracy(val, model)

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
            print(preds)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc
    
############################


####### MAIN CODE #######
USE_GPU = True
dtype = torch.float32

device = supportFunctions.getDevice(USE_GPU)

isic_data = supportClasses.ISICDataset(csv_file = 'Data/HAM10000_metadata.csv',
                        root_dir = 'Data/Images/')
    
    
#isic_data.plotRandomSample()
train, val, test = isic_data.loadDataset()

learning_rate = 1e-4

model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding = 2),
        nn.ReLU(),
        nn.Conv2d(8, 12, 3, padding = 1),
        supportClasses.Flatten(),
        nn.Linear(34968, 200),
        nn.Linear(200, 7)
)

model.apply(initWeights)
optimizer = optim.SGD(model.parameters(), lr = learning_rate,
                      momentum = 0.9, nesterov = False)

trainModel(model, optimizer, train, epochs = 5)
checkAccuracy(test, model)
############################