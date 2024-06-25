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

def train_on_dataset(ds_name, config):
    dataloaders = get_dls(ds_name, config)
    models , Xs = [], []
    model_num = 1
    for dls, splits, X, X_sw in dataloaders:
        loss_func, metrics = get_loss_func_and_metrics(config)

        # model setup
        config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
        model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
        wandbc = WandbCallback(log_preds=False, log_model=False) if config.wandb.enabled else None
        cbs = L() + wandbc
        learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)
        learn.splits = splits # This is needed for the evaluation notebook
        lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()
        
        # training 
        print("MODEL SIZE: ", get_n_params(learn), "\n")

        learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
        models.append(learn)

        Xs.append(X)

        print(f"Finished training model number:{model_num} out of {config['num']}")

        model_num += 1      
    
    evaluation(models, Xs, config)
    return models

def evaluation(models, Xs, config):
    
    for learn, X in zip(models, Xs):
        train_stats = (learn.dls.train.after_batch.mean, learn.dls.train.after_batch.std)
        ds_full = DensityData(X, lbk=config.lookback, h=config.horizon)
        tl_full = TfmdLists(range(len(ds_full)), DensityTupleTransform(ds_full))
        dl_full = TfmdDL(tl_full, bs=learn.dls.valid.bs, shuffle=False, num_workers=0, 
                    after_batch=Normalize.from_stats(*train_stats))

        n_iter = X.shape[1]//(config.horizon) - 1   
        inps, preds, targs, losses = learn.get_preds_iterative(dl=dl_full, n_iter=n_iter, track_losses=True, with_input=True)   

        for i in [inps, preds, targs, losses]:
            print(type(i))
            try:
                print(i.shape)
            except:
                print(len(i))

            print("----------"*8)

    return 

def get_datasets(ds_name, config):
    path = f'{config.data.path}{ds_name}/TLE_density_all.mat'
    data = np.array(h5py.File(path, 'r')[config["key"]])[:, :config.sel_steps]
    datasets = np.array_split(data, config['num'], axis=1) # this can result in datasets with different sizes
    data_sws = []

    for ds in datasets:
        data_sw = np.lib.stride_tricks.sliding_window_view(ds, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
        data_sw = data_sw.transpose(0,1,4,2,3)
        data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
        data_sws.append(data_sw)

    return datasets, data_sws    


def get_dls(ds_name, config):
    datasets, data_sws = get_datasets(ds_name, config)
    dataloaders = []

    for data, data_sw in zip(datasets, data_sws):
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
        dataloaders.append([dls, splits, data, data_sw])
    
    return dataloaders


if __name__ == "__main__":    
    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--ds", type = str, default = "x8x8", help = "dataset to train on")     
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--num", type = int, default = 5, help = "number of convgrus to use")
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
    for key in ['num', 'horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'downsample']:
        config[key] = arg_dict[key]

    if 100/config['num'] != 100//config['num']:
        raise "--num must divide 100 even"

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss

    print("CONFIG \n", json.dumps(config, indent=4))

    if config['downsample'] == 1:
        config_base['convgru']['n_in'] = 6
        config_base['convgru']['n_out'] = 6

    # Training
    learn = train_on_dataset(args.ds, config)