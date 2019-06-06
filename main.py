####### DEPENDENCIES #######
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

#from torch.utils.tensorboard import SummaryWriter

import numpy as np

import supportFunctions
import supportClasses
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
    loss_iter = 0
    for e in range(0, epochs):
        for t, (x, y) in enumerate(traindata):
            x = x.to(device = device, dtype = dtype)
            y = y.to(device = device, dtype = torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y, weight = lossWeights)
            
            writer.add_scalar('Cros-Entropy Loss', loss.item(), loss_iter)
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
        tboard = False
        supportFunctions.performance(TP, TN, FP, FN, printing, tboard, epoch = epoch)
############################
        
####### MAIN CODE #######
        
#Inits
USE_GPU = True
dtype = torch.float32

device = supportFunctions.getDevice(USE_GPU)
#writer = SummaryWriter()

isic_data = supportClasses.ISICDataset(csv_file = 'Data/HAM10000_metadata.csv',
                        root_dir = 'Data/Images/')
train, val, test = isic_data.loadDataset()

#Unused manual model
"""
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
"""

#Model and parameter inits
learning_rate = 5e-5
epochs = 10
model = torchvision.models.resnet101(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)
#model = nn.Sequential(nn.ReLU(), nn.Linear(256, 7))
#model.apply(initWeights)
#model = nn.Sequential(resnet, model)
#optimizer = optim.Adagrad(model.parameters(), lr = learning_rate)
start_epoch = 1

#model.apply(initWeights)
optimizer = optim.SGD(model.parameters(), lr = learning_rate,
                      momentum = 0.9, weight_decay = 0,
                      nesterov = True)


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
#images, labels = next(iter(train))
#grid = torchvision.utils.make_grid(images)
#writer.add_image('images', grid, 0)
#writer.close()

#Function calls
trainModel(model, optimizer, train, epochs = epochs, start_epoch = start_epoch)
checkPerformance(test, model, epochs + start_epoch)

#Saving
print('Do you want to save this model? [y/n] \n')
ans = input()
if ans == 'y':
    supportFunctions.saveModel(model, optimizer, epochs)
    print('Model saved succesfully')
else:
    print('Model not saved')
############################