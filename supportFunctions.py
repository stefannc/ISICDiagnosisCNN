import torch

def getDevice(USE_GPU): #Returns GPU if CUDA is available, else CPU
    
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print('Using device:', device)
    
    return device
    