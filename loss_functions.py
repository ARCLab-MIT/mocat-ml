import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *

import torch 
import torch.nn as nn


class MAELoss(nn.Module):
    def __init__(self, config):
        super(MAELoss, self).__init__()
        self.dual = config.dual

    def forward(self, y, y_hat):
        if len(y.shape) == 3:   
            y = y.unsqueeze(1)
            y_hat = y_hat.unsqueeze(1)
            
        if self.dual:
            y = y[:, :2, :, :]
            y_hat = y_hat[:, :2, :, :]

        else:
            y = y[:, :1, :, :]
            y_hat = y_hat[:, :1, :, :]
     
        absolute_diff = torch.abs(y - y_hat)
        return torch.mean(absolute_diff)


class MSELoss(nn.Module):
    def __init__(self, config):
        super(MSELoss, self).__init__()
        self.dual = config.dual

    def forward(self, pred, actual):
        if self.dual:
            y = y[:, :2, :, :]
            y_hat = y_hat[:, :2, :, :]

        else:
            y = y[:, :1, :, :]
            y_hat = y_hat[:, :1, :, :]

        squared_diff = (pred - actual) ** 2 
        return torch.mean(squared_diff)
    
    

def get_loss_function(config):
    if config.partial_loss is not None:
        if config['loss'] == 'mse':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=MSELoss(config))
        if config['loss'] == 'mae':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=MAELoss(config))

        full_loss = StackLoss()
        full_loss.__name__ = "full_loss"
        # metrics = [full_loss] 

    else:
        if config['loss'] == 'mse':
            loss_func = StackLoss(MSELoss(config))
        if config['loss'] == 'mae':
            loss_func = StackLoss(MAELoss(config))

    return loss_func




class oldSMAPE(nn.Module):
    def __init__(self, log=0, dual=0):
        super(oldSMAPE, self).__init__()
        self.log = log 
        self.dual = dual

    def forward(self, y, y_hat):
        if len(y.shape) == 3:   
            y = y.unsqueeze(1)
            y_hat = y_hat.unsqueeze(1)

        if self.dual:
            y = y[:, :2, :, :]
            y_hat = y_hat[:, :2, :, :]

        else:
            y = y[:, :1, :, :]
            y_hat = y_hat[:, :1, :, :]
       
        if self.log:
            y, y_hat = torch.exp(y)-1, torch.exp(y_hat)-1
            
        percentage_error = torch.div(torch.abs(y - y_hat), torch.abs(y) + torch.abs(y_hat)) 
        return 2*torch.mean(torch.nan_to_num(percentage_error, 0))



class totalSMAPE(nn.Module):
    def __init__(self, log=0, dual=0):
        super(totalSMAPE,self).__init__()
        self.log = log 
        self.dual = dual


    def forward(self, y, y_hat):
        if len(y.shape) == 3:   
            y = y.unsqueeze(1)
            y_hat = y_hat.unsqueeze(1)

        if self.dual:
            y = y[:, :2, :, :]
            y_hat = y_hat[:, :2, :, :]

        else:
            y = y[:, :1, :, :]
            y_hat = y_hat[:, :1, :, :]

        if self.log:
            y, y_hat = torch.exp(y)-1, torch.exp(y_hat)-1

        return 2*torch.div(torch.abs(torch.sum(y) - torch.sum(y_hat)), torch.sum(torch.abs(y)) + torch.sum(torch.abs(y_hat)))
        

class newSMAPE(nn.Module):
    def __init__(self, log=0, dual=0):
        super(newSMAPE,self).__init__()
        self.log = log 
        self.dual = dual

    def forward(self, y, y_hat):
        if len(y.shape) == 3:   
            y = y.unsqueeze(1)
            y_hat = y_hat.unsqueeze(1)

        if self.dual:
            y = y[:, :2, :, :]
            y_hat = y_hat[:, :2, :, :]

        else:
            y = y[:, :1, :, :]
            y_hat = y_hat[:, :1, :, :]

        if self.log:    
            y, y_hat = torch.exp(y)-1, torch.exp(y_hat)-1
        
        return 2*torch.div(torch.sum(torch.abs(y-y_hat)), torch.sum(torch.abs(y)) + torch.sum(torch.abs(y_hat)))



def get_metrics(config):
    metrics = [StackLoss(oldSMAPE(config.log, config.dual)), StackLoss(totalSMAPE(config.log, config.dual)), StackLoss(newSMAPE(config.log, config.dual))]
    for i, name in enumerate(["oldSMAPE", "totalSMAPE", "newSMAPE"]):
        metrics[i].__name__ = name
    return metrics



# MAE loss for ip and lr datasets
class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, y, y_hat):
        y, y_hat = y[:,:1,:,:], y_hat[:,:1,:,:]
        absolute_diff = torch.abs(y - y_hat)
        return torch.mean(absolute_diff)


def get_metrics_ip_lr(config):
    metrics = [StackLoss(oldSMAPE(config.log)), StackLoss(totalSMAPE(config.log)), StackLoss(newSMAPE(config.log))]
    for i, name in enumerate(["oldSMAPE", "totalSMAPE", "newSMAPE"]):
        metrics[i].__name__ = name
    return metrics
