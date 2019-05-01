import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import supportFunctions
import supportClasses

####### FUNCTION DEFINITIONS #######
def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

def trainModel(model, optimizer, data, epochs=1):
    model = model.to(device = device)
    traindata = data
    lossWeights = [1/262, 1/411, 1/880, 1/92, 1/890, 1/5364, 1/114]
    lossWeights = torch.tensor(np.multiply(5364, lossWeights), dtype = dtype).cuda()
    
    loss_iter = 0
    
    for e in range(0, epochs):
        for t, (x, y) in enumerate(traindata):
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y, weight = lossWeights)
            
            writer.add_scalar('Loss', loss.item(), loss_iter)
            writer.close()
            loss_iter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if (t % 20 == 0 and t != 0):
        print('Epoch', e, 'has finished')
        print('Loss:', loss.item())
        checkAccuracy(val, model)

def checkAccuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    confusion_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            #print(preds)
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            #num_correct += (preds == y).sum()
            #num_samples += preds.size(0)
        #acc = float(num_correct) / num_samples
        #print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        #return acc
        accs = 100 * confusion_matrix.diag()/confusion_matrix.sum(1)
        mean_accs = sum(accs) / 7
        print(accs)
        print('Mean:', mean_accs)
    
############################


####### MAIN CODE #######
USE_GPU = True
dtype = torch.float32

device = supportFunctions.getDevice(USE_GPU)

isic_data = supportClasses.ISICDataset(csv_file = 'Data/HAM10000_metadata.csv',
                        root_dir = 'Data/Images/')

writer = SummaryWriter()
    
    
#isic_data.plotRandomSample()
train, val, test = isic_data.loadDataset()

learning_rate = 1e-5

model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding = 2),
        nn.ReLU(),
        nn.Conv2d(8, 12, 3, padding = 1),
        nn.AvgPool2d((3,3)),
        nn.ReLU(),
        supportClasses.Flatten(),
        nn.Linear(14400, 1000),
        nn.ReLU(),
        nn.Linear(1000, 200),
        nn.ReLU(),
        nn.Linear(200, 7)
)

model = torchvision.models.resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

#model.apply(initWeights)
optimizer = optim.SGD(model.parameters(), lr = learning_rate,
                      momentum = 0.9, nesterov = True)

#TENSORBOARD CODE
images, labels = next(iter(train))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()

checkAccuracy(test, model)

print('Do you want to save this model? [y/n] \n')
ans = input()
if ans == 'y':
    supportFunctions.saveModel(model)
    print('Model saved succesfully')
else:
    print('Model not saved')
############################