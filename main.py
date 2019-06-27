####### DEPENDENCIES #######
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import supportFunctions
import supportClasses
############################

####### GLOBAL VARIABLES #######
loss_iter = 0
USE_GPU = True
dtype = torch.float32
############################

####### MAIN FUNCTIONS DEFINITIONS #######
def initWeights(m):
    """
    Currently unused. Initializes weights of a manual created model.
    """
    if type(m) == nn.Linear:
        torch.nn.init.normal(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

def trainModel(model, optimizer, data, epochs=1, start_epoch=1):
    """
    Training loop. Trains the model, calls checkPerformance() after each epoch.
    """
    #Inits
    model = model.to(device = device)
    traindata = data
    lossWeights = [1/262, 1/411, 1/880, 1/92, 1/890, 1/5364, 1/114]
    lossWeights = torch.tensor(np.multiply(5364, lossWeights), dtype = dtype).cuda()
    
    #Training loop
    global loss_iter
    for e in range(0, epochs):
        for t, (x, y) in enumerate(traindata):
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y, weight = lossWeights)
            
            writer.add_scalar('Cross-Entropy Loss', loss.item(), loss_iter)
            writer.close()
            loss_iter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch', e + 1, 'has finished')
        print('Loss:', loss.item())
        checkPerformance(val, model, e + start_epoch)

def checkPerformance(loader, model, epoch):
    """
    Checks performance of the model. Sends the data to tensorboard.
    Use "tensorboard --logdir=runs" in this folder to start the board. 
    See localhost:6006 for the board itself.
    """
    #Inits and confmat creation
    model.eval()
    confusion_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        #Calculating other statistics
        TP = confusion_matrix.diag()
        TN = []
        FP = []
        FN = []
        for c in range(7):
            idx = torch.ones(7).byte()
            idx[c] = 0
            TN.append(confusion_matrix[idx.nonzero()[:, None], idx.nonzero()].sum())
            FP.append(confusion_matrix[idx, c].sum())
            FN.append(confusion_matrix[c, idx].sum())
        
        printing = True
        tboard = True
        supportFunctions.performance(TP, TN, FP, FN, printing, tboard, epoch = epoch, writer = writer)
############################
        
####### MAIN CODE #######
        
#Inits
device = supportFunctions.getDevice(USE_GPU)
writer = SummaryWriter()

isic_data = supportClasses.ISICDataset(csv_file = 'Data/HAM10000_metadata.csv',
                        root_dir = 'Data/Images/')
train, val, test = isic_data.loadDataset()

#Model and parameter inits
learning_rate = 1e-3
epochs = 1
model = torchvision.models.resnet50(pretrained = True)
for param in model.parameters():
    #True: train full model, False: transfer learning
    param.requires_grad = False 
"""
model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 128),
        nn.BatchNorm1d(128),
        nn.Dropout(0.5),
        nn.Linear(128, 7),
        nn.ReLU())
"""
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)
start_epoch = 1

"""
optimizer = optim.SGD(model.fc.parameters(), lr = learning_rate,
                      momentum = 0.9, weight_decay = 0.01,
                      nesterov = True)
"""
optimizer = optim.Adam(model.parameters(), lr = learning_rate, amsgrad = True)

load_model = ''
if load_model == '':
    print('No pretrained loaded model found. Using default model (Resnet50)')
else:
    model, optimizer, start_epoch = supportFunctions.loadModel(model, optimizer, load_model)
    
if (USE_GPU):
    model.to(device = device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

#Tensorboard
images, labels = next(iter(train))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.close()

#Function calls
continueTraining = True
iteration = 1
while continueTraining:
    trainModel(model, optimizer, train, epochs = epochs, start_epoch = start_epoch)
    #checkPerformance(val, model, epochs + start_epoch)
    
    print(iteration * epochs, 'have ran. Do you want to run another', epochs, 'epochs? [y/n]')
    ans = input()
    if ans == 'n':
        continueTraining = False
    else:
        iteration += 1
        optimizer = optim.Adam(model.parameters(), lr = learning_rate/iteration, amsgrad = True)
        print('Learning rate has been updated from', learning_rate/(iteration-1), 'to', learning_rate/iteration)
        print('Do you want to quicksave the model? [y/n]')
        ans = input()
        if ans == 'y':
            supportFunctions.saveModel(model, optimizer, epochs)
            print('Model quicksaved succesfully. Returning to training.')
        else:
            print('Model will not be quicksaved. Returning to training.')

#Saving
print('Do you want to save this model? [y/n] \n')
ans = input()
if ans == 'y':
    supportFunctions.saveModel(model, optimizer, epochs)
    print('Model saved succesfully')
else:
    print('Model not saved')
############################