import torch
import numpy as np

def loss_func_thresholds(outputs, targets):
    ### bespoke loss_func for multi-output model
    #outputs and targets should be a batch x 8

    eps = 1e-6
    loss = -torch.sum(targets * torch.log(outputs+eps) + (1-targets)*torch.log(1-outputs+eps))
    # loss = torch.sum(torch.pow(targets - torch.sigmoid(outputs),2))
    return loss

def MSE(outputs, targets):
    return torch.sum(torch.mean((outputs - targets)**2,dim=1))

def Loss_function_poisson_adj(lamb, y,theta):
    return torch.mean(lamb - theta * y * torch.log(lamb))
