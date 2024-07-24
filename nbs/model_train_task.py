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
import json, argparse, torch, h5py
import numpy as np
from torchvision import transforms

from loss_functions import *

my_setup()
KEYS=["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]

def train_on_dataset(config):
    
    loss_func, metrics = get_loss_func_and_metrics(config)

    if config.task == 1 or config.task == 3:
        datas, data_sws = get_dataset(config)
        models = []

        for data, data_sw, key in zip(datas, data_sws, KEYS):
            dls, splits = get_dataloader(config, data, data_sw)

            config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
            model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
            wandbc = WandbCallback(log_preds=False, log_model=False) if config.wandb.enabled else None
            cbs = L() + wandbc
            learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)
            learn.splits = splits # This is needed for the evaluation notebook
            lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()
            
            # training 
            print("MODEL SIZE: ", get_n_params(learn), f"\nTraining on key: {key}")

            learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
            models.append(learn)
        return models


    if config.task == 2:
        data, data_sw = get_dataset(config)
        dls, splits = get_dataloader(config, data, data_sw)

        config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
        model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
        wandbc = WandbCallback(log_preds=False, log_model=False) if config.wandb.enabled else None
        cbs = L() + wandbc
        learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)
        learn.splits = splits # This is needed for the evaluation notebook
        lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()
        
        print("MODEL SIZE: ", get_n_params(learn), "\n")

        learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
        return learn


def get_dataloader(config, data, data_sw):
    splits = RandomSplitter()(data) #valid_pct=0.2,
    ds = DensityData(data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
    samples_per_simulation = data_sw.shape[0]//(data.shape[0])

    train_idxs = calculate_sample_idxs(splits[0], samples_per_simulation)
    valid_idxs = calculate_sample_idxs(splits[1], samples_per_simulation)

    print(len(train_idxs), len(valid_idxs))

    mocat_stats = (np.mean(data[splits[0]]), np.std(data[splits[0]]))
    train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
    valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))
    
    dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=config['bs'], device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)] if \
                        config.normalize else None,
                        num_workers=config.num_workers)
    
    return dls, splits


def get_dataset(config):  
    path = f'{config.data.path}/TLE_density_all_{config.ds}.mat'

    mat = h5py.File(path, 'r')
    datas, data_sws = [], []
    
    for key in KEYS:
        data = np.array(mat[key])[:, :config.sel_steps]

        if config.task != 3:
            num_sim, timesteps, x, y = data.shape
            data_reshaped = torch.tensor(data.reshape(-1, 1, x, y))  
            data = transforms.Resize((32, 32))(data_reshaped)  
            data = data.reshape(num_sim, timesteps, 32, 32).numpy()
            
        data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
        data_sw = data_sw.transpose(0,1,4,2,3)
        data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
        
        datas.append(data)
        data_sws.append(data_sw)

    return (np.stack(datas,2), np.stack(data_sws,2)) if config.task == 2 else (datas, data_sws)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if __name__ == "__main__":    
    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--task", type = int, default=1, help = 'task 1: independent training after resizing to 32x32 task 2: multiple channel training after resizing task 3: independent training with different shapes')
    parser.add_argument("--ds", type = str, default = "x15x15", help = "dataset to train on")     
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 4) 
    parser.add_argument("--lookback", type = int, default = 4) 
    parser.add_argument("--stride", type = int, default = 8) 
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--n_epoch", type = int, default = 20)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--loss", type = str, default = 'mse')

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['ds', 'task', 'horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps', 'loss']:
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss


    if config.task == 2:
        config_base['convgru']['n_in'] = 6
        config_base['convgru']['n_out'] = 6

    print("CONFIG \n", json.dumps(config, indent=4))

    # Training
    learn = train_on_dataset(config)

