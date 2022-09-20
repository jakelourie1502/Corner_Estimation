import torch
import numpy as np
from Prediction_functions import prediction_function_poisson, convert_from_oneHot_to_thresholds


class Acc_calc:
    def __init__(self):
        self.n = 0
        self.pos = 0
    
    def reset(self):
        self.n = 0
        self.pos = 0
    
    def get_acc(self):
        return self.pos / self.n

def train_model_thresholds(model, train_gen, val_gen, loss_func, loss_func_val, optimizer,epochs,trainable=True):
    train_acc = Acc_calc()
    val_acc = Acc_calc()
    
    for e in range(epochs):
        LOSS = 0
        model.train()
        for item in train_gen:
            x, y_1h, y_thresh  = item
            x=x.float()
            y = y_thresh
            output = model(x) #bn x 8
            y = y.float()
            loss = loss_func(output, y)

            predictions = torch.where(output>0.5, 1,0)
            correct_vals = torch.sum(predictions * y + (1-predictions)*(1-y))
            train_acc.n += len(y) * 8
            train_acc.pos += correct_vals.detach().numpy()
            if trainable:
                loss.backward()
                LOSS += loss
                optimizer.step(); optimizer.zero_grad()
        if (e+1 == epochs or (e+1)%25==0):
            print(f'==END OF EPOCH {e+1}===\n')
            print(f'Train accuracy: {train_acc.get_acc()}')
            print(f'Train loss: {LOSS / len(train_gen)}')
        train_acc.reset()
        model.eval()
        LOSS = 0
        for item in val_gen:
            x, y_1h, y_thresh  = item
            x=x.float()
            y = y_thresh
            output = model(x) #bn x 8
            loss = loss_func_val((output),y)
            LOSS += loss
            

            predictions = torch.where(output>0.5, 1,0)
            correct_vals = torch.sum(predictions * y + (1-predictions)*(1-y))
            val_acc.n += len(y) * 8
            val_acc.pos += correct_vals.detach().numpy()

        loss_val = LOSS / (val_acc.n/8)
        val_acc_final = val_acc.get_acc()
        if (e+1 == epochs or (e+1)%25==0):
            print(f'Val accuracy: {val_acc_final}')
            print(f'Val MSE: {loss_val}')
        val_acc.reset()
    return loss_val, val_acc_final

def train_model_softmax(model, train_gen, val_gen, loss_func, loss_func_val, optimizer,epochs):
    train_acc = Acc_calc()
    val_acc = Acc_calc()
    train_acc_thresh = Acc_calc()
    val_acc_thresh = Acc_calc()
    for e in range(epochs):
        LOSS = 0
        model.train()
        for item in train_gen:
            x, y_1h, y_thresh  = item
            x=x.float()
            y = y_1h.float()
            y_thresh=y_thresh.numpy()
            
            #calc loss
            output = model(x) #bn x 8
            
            loss = loss_func(output, y)
            
            #calc predictions for individual corner amounts
            predictions = torch.argmax(output,1)
            y = torch.argmax(y,1)
            correct_vals = torch.sum(torch.where(predictions == y, 1,0))
            train_acc.n += len(y)
            train_acc.pos += correct_vals.detach().numpy()
            
            #calculate "accuracy"
            threshold_values = convert_from_oneHot_to_thresholds(output)
            predictions = torch.where(threshold_values>0.5, 1,0)
            correct_vals = torch.sum(predictions * y_thresh + (1-predictions)*(1-y_thresh))
            train_acc_thresh.pos += correct_vals.detach().numpy()
            train_acc_thresh.n += len(y) * 8
            
            loss.backward()
            LOSS += loss
            optimizer.step(); optimizer.zero_grad()
        if (e+1) % 20 == 0:
            print(f'==END OF EPOCH {e+1}===\n')
            print(f'Train accuracy specific number of corners: {train_acc.get_acc()}')
            print(f'Train accuracy thresh: {train_acc_thresh.get_acc()}')
            print(f'Train loss: {LOSS / (train_acc_thresh.n/8)}')
        train_acc.reset()
        train_acc_thresh.reset()
        model.eval()
        LOSS = 0
        with torch.no_grad():
            for item in val_gen:
                x, y_1h, y_thresh  = item
                x=x.float()
                y = y_1h.float()
                y_thresh=y_thresh.numpy()

                #calc loss
                output = model(x) #bn x 8
                loss = loss_func(output, y)
                
                #calc predictions for individual corner amounts
                predictions = torch.argmax(output,1)
                y = torch.argmax(y,1)
                correct_vals = torch.sum(torch.where(predictions == y, 1,0))
                val_acc.n += len(y)
                val_acc.pos += correct_vals.detach().numpy()

                #calculate "accuracy"
                threshold_values = convert_from_oneHot_to_thresholds(output)
                predictions = torch.where(threshold_values>0.5, 1,0)
                correct_vals = torch.sum(predictions * y_thresh + (1-predictions)*(1-y_thresh))
                val_acc_thresh.pos += correct_vals.detach().numpy()
                val_acc_thresh.n += len(y) * 8
                loss = loss_func_val(threshold_values,y_thresh)
                LOSS += loss

            loss_val = LOSS / (val_acc_thresh.n/8)
            val_acc_final = val_acc.get_acc()
            val_acc_thresh_final = val_acc_thresh.get_acc()
            if (e+1) % 20 == 0:
                print(f'Val accuracy specific number of corners: {val_acc_final}')
                print(f'Val acc thresh: {val_acc_thresh_final}')
                print(f'Val MSE: {loss_val}')
            val_acc.reset()
            val_acc_thresh.reset()
    return loss_val, val_acc_final


def train_model_poisson(model, train_gen, val_gen, loss_func, loss_func_val, optimizer,epochs,theta=False,max_val=30):
    
    train_acc = Acc_calc()
    val_acc = Acc_calc()
    train_acc_thresh = Acc_calc()
    val_acc_thresh = Acc_calc()
    for e in range(epochs):
        LOSS = 0
        model.train()
        
        for item in train_gen:
            x, y_1h, y_thresh = item
            x=x.float()
            y = torch.argmax(y_1h,1).float().reshape(-1,1) #converts one hot to a single value.
            y_thresh=y_thresh.numpy()
            
            #calc loss
            output = model(x) #bn x 1 (one value of lambda for each item in the batch)
            # if theta == False: loss = loss_func(output, y)
            # else: loss = loss_func(output, y, theta)
            loss = loss_func(output, y)
            
            #calc predictions for individual corner amounts
            predictions, argmax = prediction_function_poisson(output, max_val, theta)
            correct_vals = torch.sum(torch.where(argmax == y.reshape(-1), 1,0))
            train_acc.n += len(y)
            train_acc.pos += correct_vals.detach().numpy()
            
            #calculate "accuracy"
            threshold_values = convert_from_oneHot_to_thresholds(predictions,sm=False)
            predictions = torch.where(threshold_values>0.5, 1,0)
            correct_vals = torch.sum(predictions * y_thresh + (1-predictions)*(1-y_thresh))
            train_acc_thresh.pos += correct_vals.detach().numpy()
            train_acc_thresh.n += len(y) * 8
            
            loss.backward()
            LOSS += loss
            optimizer.step(); optimizer.zero_grad()
        
        if (e+1) % 25 == 0:
            print(f'==END OF EPOCH {e+1}===\n')
            print(f'Train accuracy specific no. of corners : {train_acc.get_acc()}')
            print(f'Train accuracy thresh: {train_acc_thresh.get_acc()}')
            print(f'Train loss: {LOSS / (train_acc_thresh.n/8)}')
        train_acc.reset()
        train_acc_thresh.reset()
        model.eval()
        LOSS = 0
        with torch.no_grad():
            for item in val_gen:
                x, y_1h, y_thresh = item
                x=x.float()
                y = torch.argmax(y_1h,1).float().reshape(-1,1) #converts one hot to a single value.
                y_thresh=y_thresh.numpy()

                #calc loss
                output = model(x) #bn x 8
                
                #calc predictions for individual corner amounts
                predictions, argmax = prediction_function_poisson(output, max_val, theta)
                correct_vals = torch.sum(torch.where(argmax == y.reshape(-1), 1,0))
                val_acc.n += len(y)
                val_acc.pos += correct_vals.detach().numpy()

                #calculate "accuracy"
                threshold_values = convert_from_oneHot_to_thresholds(predictions, sm=False)
                predictions = torch.where(threshold_values>0.5, 1,0)
                correct_vals = torch.sum(predictions * y_thresh + (1-predictions)*(1-y_thresh))
                val_acc_thresh.pos += correct_vals.detach().numpy()
                val_acc_thresh.n += len(y) * 8
                loss = loss_func_val(threshold_values,y_thresh)
                LOSS += loss

            loss_val = LOSS / (val_acc_thresh.n/8)
            val_acc_final = val_acc.get_acc()
            val_acc_thresh_final = val_acc_thresh.get_acc()
            if (e+1) % 25 == 0:
                print(f'Val specific no. of corners: {val_acc_final}')
                print(f'Val acc thresh: {val_acc_thresh_final}')
                print(f'Val MSE: {loss_val}')
            val_acc.reset()
            val_acc_thresh.reset()
    return loss_val, val_acc_final
