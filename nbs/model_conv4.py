import sys, os
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

from mocatml.utils import *
# convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from tsai.imports import my_setup
from tsai.utils import yaml2dict, dict2attrdict
from fastai.callback.schedule import valley, steep
from mygrad import sliding_window_view

import wandb, json, argparse, torch, time, math
import numpy as np

my_setup()
from utils import *
from mocatml.models.conv4_rnn import *

from fastai.vision.all import *
from fastai.distributed import *


def train_model_4d(config):
    dls, splits, _, _ = get_dataloader_4d(config)
    loss_func, metrics = get_loss_func_and_metrics(config)

    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = Stack4Unstack(Simple4Model(**config.convgru))
    
    cbs = L() 
    learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)
    learn.splits = splits # This is needed for the evaluation notebook
    lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()
    
    # training 
    print("MODEL SIZE: ", get_n_params(learn), "\n")
    # with learn.distrib_ctx(sync_bn=False):
    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)

    print(learn.recorder.losses)


if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--ds", type = str, default = "x8x8", help = "dataset to train on")     
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 4) 
    parser.add_argument("--lookback", type = int, default = 4) 
    parser.add_argument("--stride", type = int, default = 8) 
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--n_epoch", type = int, default = 20)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--loss", type = str, default = 'mae')
    parser.add_argument("--num_workers", type = int, default=0)

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['ds', 'horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps', 'loss', 'num_workers']:
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss

    print("CONFIG \n", json.dumps(config, indent=4))

    # model setup
    train_model_4d(config)

