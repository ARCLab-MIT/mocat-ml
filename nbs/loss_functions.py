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

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y, y_hat):
        absolute_diff = torch.abs(y - y_hat)
        percentage_error = torch.div(absolute_diff, torch.clamp(torch.abs(y), min=self.eps))
        return torch.mean(percentage_error)
    

class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y, y_hat):
        percentage_error = torch.div(torch.abs(y - y_hat), torch.abs(y) + torch.abs(y_hat)) 
        return torch.mean(torch.nan_to_num(percentage_error, 1))

class MBDLoss(nn.Module):
    def __init__(self, alpha):
        super(MBDLoss, self).__init__()
        self.mse = nn.MSELoss()  # Mean squared error loss
        self.mae = nn.L1Loss()   # Mean absolute error loss
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        mae_loss = self.mae(y_pred, y_true)

        mbd_loss = (1 - self.alpha) * mse_loss + self.alpha * mae_loss
        return mbd_loss

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


def get_loss_func_and_metrics(config):
    if config.partial_loss is not None:
        if config['loss'] == 'mse':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=MSELossFlat())
        if config['loss'] == 'mae':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=L1LossFlat())
        if config['loss'] == 'huber':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=nn.HuberLoss())
        if config['loss'] == 'mbd':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=MBDLoss(config['alpha']))
        if config['loss'] == 'mape':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=MAPELoss())
        if config['loss'] == 'rmsle':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=RMSLELoss())
        if config['loss'] == 'smape':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=SMAPELoss())

        full_loss = StackLoss()
        full_loss.__name__ = "full_loss"
        metrics = [full_loss] 

    else:
        if config['loss'] == 'mse':
            loss_func = StackLoss(MSELossFlat())
        if config['loss'] == 'mae':
            loss_func = StackLoss(L1LossFlat())
        if config['loss'] == 'huber':
            loss_func = StackLoss(nn.HuberLoss())
        if config['loss'] == 'mbd':
            loss_func =  StackLoss(MBDLoss(config['alpha']))
        if config['loss'] == 'mape':
            loss_func =  StackLoss(MAPELoss())
        if config['loss'] == 'rmsle':
            loss_func = StackLoss(RMSLELoss())

    if config['metric'] == 'smape':
        metrics = [StackLoss(SMAPELoss())]
        metrics[0].__name__ = "SMAPE"
    return loss_func, metrics