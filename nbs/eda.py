import h5py, os
import numpy as np
import matplotlib.pyplot as plt

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

def print_ds(ds_name):
    path = f"/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/{ds_name}/TLE_density_all.mat"
    data = h5py.File(path, 'r')
    for key in data.keys():
        try:
            print(key, data[key].shape)
        except:
            print(key, type(data[key]))
        pass

def count_nonzeros_over_time(data, stride):
    num_timesteps = data.shape[0]
    zeros_per_timestep = []
    for t in range(0, num_timesteps, stride):
        zeros_per_timestep.append(np.count_nonzero(data[t]))
    return zeros_per_timestep


def plot_nonzeros(ds_name, key, data, stride=8):
    plt.figure(figsize=(8, 6))
    count = count_nonzeros_over_time(data, stride)
    print(len(count))
    plt.plot(np.linspace(0, data.shape[0], len(count)), count)
    plt.xlabel("Timestep")
    plt.ylabel("Number of Non-Zeros")
    plt.title("Number of Non-Zeros Over Time")
    plt.grid(True)
    path = f'plots_eda/{key}'
    if not os.path.exists(path):
        os.makedirs(path)
        
    plt.savefig(f"plots_eda/{key}/{ds_name}_{key}_num_of_nonzeros")
    plt.show()
    plt.close()

if __name__ == "__main__":

    model_type = 'convgru'
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)


    data = np.load(Path('~/mocat-ml/data/TLE_density_all_x15x15.npy').expanduser(), 
               mmap_mode='c' if config.mmap else None)  

    data = data[0]
    print(data.shape)
