import torch
import numpy as np 
import torch.nn as nn
def x_Y_transform(x, Y, max_val):
    """Takes the number of corners y[2] (currently don't use home and away corners)
    outputs 9 objects:
    1. One hot encoding (with start end points at 5 and 15)
    2-9 (below/above * 9-12) # produced both in dictionary form and tensor form
    x values in tensor
    """
    y = [x[2] for x in Y]
    def one_hot_func(y,max_val):
        Y_1h = nn.functional.one_hot(torch.tensor(np.array(y)),max_val)
        return Y_1h
            
    
    def produce_binaries(threshold,y):
        #print(f'Threshold: {threshold}, y: {y}')
        y_below, y_above =[], []
        for tgt in y:
            #print("NEW", tgt);
            if tgt < threshold: 
                y_below.append(1); #print("ybelow1")
            else: 
                y_below.append(0); #print("ybelow0") 
            if tgt > threshold:
                y_above.append(1); #print("yabove1")
            else:
                y_above.append(0); #print("yabove0")
        return y_below, y_above
    
    thresh_targets = {}
    for thr in [9, 10, 11, 12]:
        thresh_targets[thr] = produce_binaries(thr,y)
        
    y_tensor = []
    for tgt in [9,10,11,12]:
        y_tensor.append(thresh_targets[tgt][0]); y_tensor.append(thresh_targets[tgt][1])
    y_tensor = torch.tensor(np.array(y_tensor)).T
    x_tensor = torch.tensor(np.array(x))
    
    return one_hot_func(y,max_val), y_tensor, thresh_targets, x_tensor

class Dataset(torch.utils.data.Dataset):

  def __init__(self, x, y_1h, y_thresh): #requires tensor inputs
        self.y_1h = y_1h
        self.y_thresh = y_thresh
        self.x = x

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.y_thresh)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.x[index], self.y_1h[index], self.y_thresh[index]
    