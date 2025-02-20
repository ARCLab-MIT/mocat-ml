import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
from tsai.imports import my_setup
from tsai.utils import yaml2dict
from fastai.callback.wandb import WandbCallback
import wandb, json, argparse, os, torch, random, datetime
import numpy as np

from loss_functions import *
from convgru_utils import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


my_setup()


def train(ds_list, config):
    # data setup
    dls, splits, Xs  = get_dls(ds_list, config)
    loss_func, metrics = get_loss_function(config), get_metrics(config)

    # model setup
    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
    learn = Learner(dls, model, loss_func=loss_func, cbs=[], metrics=metrics)
    learn.splits = splits # This is needed for the evaluation notebook
    lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()


    # Training
    print("MODEL SIZE: ", get_n_params(learn), "\n")
    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    print("Training DONE!")


    # Directories to save results 
    date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())
    os.makedirs(f"results/{date}")

    # Save training history
    save_training_metrics(learn, metrics, save_folder=f'results/{date}/')

    # Save config as txtfile
    with open(f"results/{date}/config.txt", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Evalution
    model.eval()
    for X, split, ds in zip(Xs, splits, ds_list):

        ind = random.choice(split[1])
        val = torch.tensor(X[ind])

        # Iterative forecasting
        initial_seq = torch.tensor(val[:config.lookback], device=default_device()).float()
        inp = tuple(i.reshape(1,1,32,32) for i  in initial_seq)

        n_iter = 2436//config.lookback
        preds = [initial_seq]
        for _ in range(n_iter):
            p = model(inp)
            preds.append(torch.squeeze(torch.vstack(p), dim=1))
            inp = p

        preds = torch.vstack(preds)[:2436].cpu().detach()
        plot_save_results(val, preds, config, ds, date, metrics)

    # Save model
    if config.save_model:
        torch.save(learn.model.state_dict(), f"results/{date}/model.pth")


    # TODO script to produce a grid of smape values across ip/lr/pmd/cam

if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")


    # Data settings 
    parser.add_argument("--init_pop", nargs="+", type=int, default = [1, 1])   # range of initial populations to use for training
    parser.add_argument("--launch_rate", nargs="+", type=int, default = [2, 2])  # range of launch rates to use for training 
    parser.add_argument("--pmd", nargs="+", type=float, default=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99])
    parser.add_argument("--cam", nargs="+", type=float, default=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99])
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--lookback", type = int, default = 50)
    parser.add_argument("--gap", type = int, default = 0) 
    parser.add_argument("--stride", type = int, default = 100) 
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 16)
    parser.add_argument("--n_epoch", type = int, default = 30)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--loss", type = str, default = 'mae')
    parser.add_argument("--log", type = int, default = 1)  # whether or not to train on log(N)
    parser.add_argument("--average", type = int, default = 1) # whether or not to use averaging 
    parser.add_argument("--norm", type = int, default = 1) # whether or not to normalize (Z-norm) during training
    parser.add_argument("--save_model", type = int, default = 0) # whether or not to save the model


    # Set defaults 
    args = parser.parse_args()
    args.init_pop = [i for i in range(args.init_pop[0], args.init_pop[1]+1)]    
    args.launch_rate = [i for i in range(args.launch_rate[0], args.launch_rate[1]+1)]   
    args.ds_list = [f'x{init_pop}x{launch_rate}x{pmd:.2f}x{cam:.2f}' for init_pop in args.init_pop for launch_rate in args.launch_rate for pmd in args.pmd for cam in args.cam] # Default datasets to train on
    arg_dict = vars(args)


    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    config.partial_loss = [0] if args.partial_loss == 1 else None

    keys = ['ds_list', 'launch_rate', 'init_pop', 'pmd', 'cam', 'horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'log', 'average', 'norm', 'save_model']
    for key in keys:
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss

    print("CONFIG \n", json.dumps(config, indent=4))

    # Training
    train(args.ds_list, config)   


