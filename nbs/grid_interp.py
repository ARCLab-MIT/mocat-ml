import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
from mygrad import sliding_window_view
from tsai.imports import my_setup
from tsai.utils import yaml2dict, dict2attrdict
from fastai.callback.schedule import valley, steep
from fastai.callback.wandb import WandbCallback
import wandb, json, argparse, os, h5py

my_setup()

def get_dataset(dataset, config):
    path = f'~/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/{dataset}/TLE_density_all.mat'
    data = h5py.File(path, 'r')
    data = np.array(data['comb_Am_rp'][:])

    data = data[:, :config.sel_steps]
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    samples_per_simulation = data_sw.shape[1]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw


def get_val_dls(config):
    data, data_sw = None, None
    for xPop in range(config['xPop_1'], config['xPop_2']+1):
        for xLaunch in range(config['xLaunch_1'], config['xLaunch_2']+1):
            if xPop + xLaunch == 0: continue

            ds_name = f'x{xPop}x{xLaunch}'
            if data is None: 
                data, data_sw = get_dataset(ds_name, config)
            else:

                ds, ds_dw = get_dataset(ds_name, config)
                data = np.vstack([data, ds])
                data_sw = np.vstack([data_sw, ds_dw])

    print(data.shape, data_sw.shape)

if __name__ == "__main__":
    
    # TODO - add wandb implementation and more architectures

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--ds_grid", type = str, default = "5,10,5,10", help = "grid datasets to train on") 
    parser.add_argument("--interp", type = bool, default = True)    
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 4)
    parser.add_argument("--lookback", type = int, default = 4)
    parser.add_argument("--stride", type = int, default = 8)
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--n_epoch", type = int, default = 20)
    parser.add_argument("--sel_steps", type = int, default = None)

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps']:
        config[key] = arg_dict[key]

    ranges = args.ds_grid.split(',')
    if len(ranges) != 4:
        raise ValueError("Wrong format")
    
    for i, j in zip(ranges, ['xPop_1', 'xPop_2', 'xLaunch_1', 'xLaunch_2']):
        config[j] = int(i)

    print("CONFIG \n", json.dumps(config, indent=4))