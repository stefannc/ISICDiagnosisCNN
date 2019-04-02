import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import supportFunctions
import supportClasses

####### FUNCTION DEFINITIONS #######

def trainModel(model, optimizer, epochs=1):
    
    model = model.to(device = device)
    
    for e in range(0, epochs):
        for t, (x, y) in enumerate(train):
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            print('Loss:', loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (t % 5 == 0):
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

learning_rate = 1e-2

model = nn.Sequential(
        nn.Conv2d(3, 8, 5, padding = 2),
        nn.ReLU(),
        supportClasses.Flatten(),
        nn.Linear(21600, 7)
)

optimizer = optim.SGD(model.parameters(), lr = learning_rate,
                      momentum = 0.9, nesterov = True)

trainModel(model, optimizer)
checkAccuracy(test, model)

############################