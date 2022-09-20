import torch
import numpy as np
from math import factorial as fac
import torch.nn as nn

def poissonPDF(lamb, k):
    return (lamb**k) * np.exp(-lamb) / fac(int(k))

def poissonPDF_adj(lamb, k,theta):
    return np.exp(-lamb) *((lamb**k) / fac(int(k)))**(theta) + np.max([0,lamb-k])*6e-5

def prediction_function_poisson(lamb, max_val, theta=False,simulator=False):
    if not simulator: lamb = lamb.detach().cpu().numpy().squeeze()
    else: lamb = lamb.detach().cpu().numpy()
    """we take in lamb and predict the likelihood of each corners (1) and the argmax of those (2)"""
    if theta == False: #normal poisson
        predictions = np.array([[poissonPDF(l, k) for k in range(max_val)] for l in lamb])
    else:
        predictions = np.array([[poissonPDF_adj(l, k, theta) for k in range(max_val)] for l in lamb])
        predictions = predictions / np.sum(predictions,axis=1,keepdims=True)
    argmax = np.argmax(predictions,1)
    return torch.tensor(predictions), torch.tensor(argmax)

def convert_from_oneHot_to_thresholds(x,sm=True):
    if sm: x = nn.Softmax(dim=1)(x)
    
    nine_idx, ten_idx, eleven_idx, twelve_index = 9, 10, 11, 12
    below_9 = torch.sum(x[:,:nine_idx],dim=1)
    above_9 = torch.sum(x[:,nine_idx+1:],dim=1)
    below_10 = torch.sum(x[:,:ten_idx],dim=1)
    above_10 = torch.sum(x[:,ten_idx+1:],dim=1)
    below_11 = torch.sum(x[:,:eleven_idx],dim=1)
    above_11 = torch.sum(x[:,eleven_idx+1:],dim=1)
    below_12 = torch.sum(x[:,:twelve_index],dim=1)
    above_12 = torch.sum(x[:,twelve_index+1:],dim=1)
    
    # print(below_9)
    return torch.stack((below_9, above_9, below_10, above_10, below_11, above_11, below_12, above_12),dim=1)
    