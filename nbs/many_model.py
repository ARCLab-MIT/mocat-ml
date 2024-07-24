import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
from tsai.imports import my_setup
from tsai.utils import yaml2dict, dict2attrdict
from fastai.callback.schedule import valley, steep
from mygrad import sliding_window_view
from fastai.callback.wandb import WandbCallback
import wandb, json, argparse, os, torch
import numpy as np

from loss_functions import *
from utils import *

my_setup()

# much faster on workstation
def train_on_dataset(ds_name, config):
    # only implemented for convgru (add more architectures)
    dls, X, X_sw = get_dataloader(ds_name, config)

    loss_func, metrics = get_loss_func_and_metrics(config)

    # model setup
    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
    wandbc = WandbCallback(log_preds=False, log_model=False) if config.wandb.enabled else None
    cbs = L() + wandbc
    learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)
    # learn.splits = splits # This is needed for the evaluation notebook
    # lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()


    
    # # training 
    # print("MODEL SIZE: ", get_n_params(learn), "\n")

    # learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    # # learn.fit(config.n_epoch, 1e-3)
    if config.partial_loss is None:
        save_folder = f"plots/{config['loss']}/{ds_name}_stride_{config['stride']}/" if config['loss'] != 'mbd' else f"plots/{config['loss']}/{ds_name}_stride_{config['stride']}/alpha_{config['alpha']}/"
    else:
        save_folder = f"plots/{config['loss']}_partial/{ds_name}_stride_{config['stride']}/" if config['loss'] != 'mbd' else f"plots/{config['loss']}_partial/{ds_name}_stride_{config['stride']}/alpha_{config['alpha']}/"

    plot_preds(learn, config, X, X_sw, save_folder)
    return learn



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
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--loss", type = str, default = 'mse')
    parser.add_argument("--downsample", type = int, default = 0)

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'downsample']:
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss

    print("CONFIG \n", json.dumps(config, indent=4))

    if config['downsample'] == 1:
        config_base['convgru']['n_in'] = 6
        config_base['convgru']['n_out'] = 6


    

    # Training
    learn = train_on_dataset(args.ds, config)


    # # Loss plot
    # path = f'plots/{args.dataset}/stride_{config.stride}_bs_{config.bs}/num_epochs_{config.n_epoch}/'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # for i in [0, 0.5, 0.75, 0.9]:
    #     num_epochs_toshow = config.n_epoch - int(i*config.n_epoch)
    #     fig, ax = plt.subplots()
    #     skip_start = int(len(learn.recorder.losses) * i)
    #     plot_loss(learn.recorder, skip_start=skip_start, ax=ax)
    #     ax.set_title('learning curve full' if skip_start == 0 else f'learning curve last {num_epochs_toshow} epochs')
    #     name = 'full' if i==0 else f'last {num_epochs_toshow} epochs'
    #     plt.savefig(f'{path}{name}.png')
    #     plt.show()

