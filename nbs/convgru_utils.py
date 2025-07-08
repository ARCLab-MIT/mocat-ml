import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
import os, h5py, torch
import numpy as np

import torch
from torchvision import transforms

from loss_functions import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset


import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import math

import pandas as pd



KEYS=["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]

def calculate_sample_idxs(simulation_idxs, samples_per_sim):
    indices = []
    for sim in simulation_idxs:
        start_idx = sim * samples_per_sim
        end_idx = start_idx + samples_per_sim
        indices.extend(range(start_idx, end_idx))
    return indices



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def make_and_save_gif(target, preds, config, savename, stride=10):
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))  # Create a figure with three subplots
    preds, target, abs_diff = preds+1, target+1, torch.abs(target-preds)+1 # Adding one to avoid error when putting log scale 
    
    vmin, vmax = 1, max([torch.max(target).item(), torch.max(preds).item()])
    log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    extent = [6578.137, 8378.136999999999, 9.999999999999999e-06, 10.0]
    
    # Create images for the first and second distributions
    im1 = ax1.imshow(target[0, :, :], aspect='auto', norm=log_norm, extent = extent)
    im2 = ax2.imshow(preds[0, :, :], aspect='auto', norm=log_norm, extent = extent)
    im3 = ax3.imshow(abs_diff[0, :, :], aspect='auto', norm=log_norm, extent = extent)    

    # Add colorbars
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    # Add titles
    ax1.set_title("MOCAT-MC")
    ax2.set_title("MOCAT-ML")
    ax3.set_title("Absolute Difference")
    ax4.set_title("Model Configuration")

    # Add x and y labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("rp [km]")
        ax.set_ylabel("Am [m2/kg]")

    # Display model configuration in the third subplot
    model_config = {key:config[key] for key in ['launch_rate', 'init_pop', 'pmd', 'cam', 'horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'key', 'loss', 'log', 'average']}
    config_text =  "\n\n" + "\n".join([f"{key}: {value}" for key, value in model_config.items()])
    ax4.text(0.05, 0.3, config_text, fontsize=8) 
    ax4.axis('off') 

    n_iter = 2436//(config.lookback  + config.gap) - math.ceil(config.lookback/(config.lookback + config.gap))

    def animate(i):
        im1.set_array(target[i * stride, :, :])
        im2.set_array(preds[i * stride, :, :])
        im3.set_array(abs_diff[i * stride, :, :])

        # Update the title with the current iteration and year
        fig.suptitle(f'Forecast iteration {round(i * stride / 2436 * n_iter)} year: {round(i * stride / 2436 * 100)}')
        return im1, im2, im3, fig

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=target.shape[0] // stride, interval=20, blit=True)

    # Save the animation as a GIF
    ani.save(savename, writer='imagemagick', fps=20)
    plt.close()


def get_ds(ds_name, config):  

    data_path = "/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam"
    path = f'{data_path}/TLE_density_all_{ds_name}.mat'
    mat = h5py.File(path, 'r')
    data = np.array(mat[config.key])[:, :config.sel_steps]
    _, timesteps, w, h, c = data.shape
    data = data.sum(axis=-1)

    if config.average:
        data = data.reshape(5, -1, timesteps, w, h).mean(axis=1)

    if config.log:
        data = np.log(data + 1)

    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw


def get_dls(ds_list, config):

    train_dls, valid_dls, splits, Xs = [], [], [], []
    for ds_name in ds_list:
        X, X_sw = get_ds(ds_name, config)
        split = RandomSplitter()(X) #valid_pct=0.2,
        ds = DensityData(X_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
        
        samples_per_simulation = X_sw.shape[0]//(X.shape[0])
        train_idxs = calculate_sample_idxs(split[0], samples_per_simulation)
        valid_idxs = calculate_sample_idxs(split[1], samples_per_simulation)

        print("Loaded dataset: ", ds_name, " Shapes: ", X.shape, X_sw.shape)

        train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
        valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))

        train_dls.append(train_tl)
        valid_dls.append(valid_tl)
        splits.append(split)
        Xs.append(X)

    train = np.concatenate([X[split[0]] for X, split in zip(Xs, splits)], axis=0)   
    mocat_stats = (np.mean(train), np.std(train))

    train, valid = ConcatDataset(train_dls), ConcatDataset(valid_dls)
    dls = DataLoaders.from_dsets(train, valid, bs=config.bs, device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)] if \
                        config.norm else None,
                        num_workers=config.num_workers)

    return dls, splits, Xs




def save_training_metrics(learn, metrics, save_folder):

    history = learn.recorder.values
    metric_names = [m.__name__ for m in metrics]
    columns = ['train_loss', 'valid_loss'] + metric_names
    df = pd.DataFrame(history, columns=columns)

    # Save to CSV
    df.to_csv(f"{save_folder}training_metrics.csv", index=False)    


    # Extract losses and metrics
    epochs = range(1, len(history) + 1)
    train_loss = [row[0] for row in history]
    valid_loss = [row[1] for row in history]
    metric_values = list(zip(*[row[2:] for row in history]))  # Unzips metrics for separate plotting

    # Plot Training & Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o', linestyle='-', color='blue')
    plt.plot(epochs, valid_loss, label="Valid Loss", marker='o', linestyle='-', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("MAE Loss")
    plt.title("Training & Validation Loss Evolution")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_folder}train_val_loss.png")

    # Plot All Metrics Together
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metric_names):
        plt.plot(epochs, metric_values[i], label=metric, marker='o', linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Metrics Evolution")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_folder}metrics.png")




def plot_save_results(val, preds, config, ds, date, metrics):

    # Directory to save results
    os.makedirs(f"results/{date}/{ds}")
    years = np.linspace(0, 100, 2436)[config.lookback:]

    # Plotting metrics over time 
    for smape in metrics:
        smapes = [smape.loss_func(i, j).item() for i,j in zip(preds[config.lookback:], val[config.lookback:])]
        fig = plt.figure()
        fig.clf()
        plt.plot(years, smapes, label=f'ds: {ds}')
        plt.xlabel("Years")
        plt.title(f"{smape.__name__} over time")
        plt.legend()
        plt.savefig(f'results/{date}/{ds}/{smape.__name__}.png')
        plt.show()

    if config.log:
        val, preds = torch.exp(val)-1, torch.exp(preds)-1

    # Plot loss over time 
    maes = [L1LossFlat()(i, j).item() for i, j in zip(preds[config.lookback:], val[config.lookback:])]
    fig = plt.figure()
    fig.clf()
    plt.plot(years, maes, label=f'Losses ds: {ds}')
    plt.title("Mean absolute error over time")
    plt.xlabel("Years")
    plt.legend()
    plt.savefig(f'results/{date}/{ds}/losses.png')
    plt.show()


    # Total number of objects over time
    mc, ml = val.sum(dim=(-1,-2))[config.lookback:], preds.sum(dim=(-1,-2))[config.lookback:]
    fig = plt.figure()
    fig.clf()
    plt.plot(years, mc, label=f'MC ds: {ds}')
    plt.plot(years, ml, label=f'ML ds: {ds}')
    plt.title("Total number of objects over time")
    plt.xlabel("Years")
    plt.legend()
    plt.savefig(f'results/{date}/{ds}/total_num_objects.png')
    plt.show()

    # Produce gif for the prediction
    make_and_save_gif(val, preds, config, f"results/{date}/{ds}/forecast.gif")


