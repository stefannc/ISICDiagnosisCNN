import torch

def getDevice(USE_GPU): #Returns GPU if CUDA is available, else CPU
    
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print('Using device:', device)
    
    return device

def saveModel(model, optimizer, epochs):
    state = {'epoch': epochs + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, 'checkpoint.pth')
    return

def loadModel(model, optimizer, filename):
    print('Loading checkpoint from', filename)
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Loaded checkpoint at epoch:', start_epoch)
    
    return model, optimizer, start_epoch
    
