import torch
import numpy as np

def getDevice(USE_GPU): #Returns GPU if CUDA is available, else CPU
    
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print('Using device:', device)
    
    return device

def sigmoidConversion(x, a = 1, threshold = 0.5):
    x = x.cpu()
    return (1/(1 + np.exp(-1 * a * (x - threshold))))

def saveModel(model, optimizer, epochs):
    state = {'epoch': epochs + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, 'checkpoint.pth')
    return

def loadModel(model, optimizer = None, filename = None):
    if not filename == None:
        print('Loading checkpoint from', filename)
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not optimizer == None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded checkpoint at epoch:', start_epoch)
    
    return model, optimizer, start_epoch

def performance(TP, TN, FP, FN, printing, tboard, epoch = 0, writer = None):
    class_names = ['melanoma', 'nevus', 'carcinoma', 'bowens', 'keratosis', 'dermatofibroma', 'vascular']
    accuracy = []
    sensitivity = []
    specificity = []
    precision = []
    F1Score = []
    for i in range(7):
        accuracy.append(100*(TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] + FN[i]))
        sensitivity.append(100*(TP[i])/(TP[i]+FN[i]))
        specificity.append(100*(TN[i])/(TN[i]+FP[i]))
        precision.append(100*(TP[i])/(TP[i]+FP[i]))
        F1Score.append((2*TP[i])/(2*TP[i] + FP[i] + FN[i]))
        
        
        if printing:
            print('########################################################')
            print(class_names[i])
            print('Accuracy:', accuracy[i])
            print('Sensitivity/Recall:', sensitivity[i])
            print('Specificity', specificity[i])
            print('Precision', precision[i])
            print('F1 Score', F1Score[i])
            print('########################################################')
                  
    m_acc = np.sum(accuracy)/7
    m_rec = np.sum(sensitivity)/7
    m_pre = np.sum(precision)/7
    m_f1 = np.sum(F1Score)/7
    if printing:
        print('########################################################')
        print('Mean Accuracy:', m_acc)
        print('Mean Recall:', m_rec)
        print('Mean Precision:', m_pre)
        print('Mean F1-Score:', m_f1)
        print('########################################################')
    if tboard:
        writer.add_scalar('Accuracy', m_acc, epoch)
        writer.add_scalar('Recall', m_rec, epoch)
        writer.add_scalar('Precision', m_pre, epoch)
        writer.add_scalar('F1-Score', m_f1, epoch)
        writer.close()

