import sys, json, os
sys.path.append('..')
from tsai.utils import yaml2dict
import argparse, torch, h5py, torchvision, datetime

from fastcore.all import *
from fastai.vision.all import *

from tqdm import tqdm 
import matplotlib.colors as mcolors
from nbs.utils import *
from loss_functions import *


DATA_PATH = "/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam" #TODO change it accordingly
N, KEY, LKB = 2436, 'comb_Am_rp', 50
NUMBER_SIZE, FONT_SIZE, TITLE_SIZE = 20, 20, 26


# Load data
def get_data(ds):
    path = f'{DATA_PATH}/TLE_density_all_{ds}.mat'
    mat = h5py.File(path, 'r')
    data = np.array(mat[KEY]).copy()
    data = np.transpose(data, (0, 1, 4, 2, 3))
    return data


#TODO adjust model path 
def get_pred(cfg, ds):
    """
    Forecasts using a given model 
    Always uses guided model with horizon 50
    guide=1 means model takes guidance where guidance is added as additional channels
    dual=1 means model returns active/ianctive objects as 2 channels
    """
    dual, guide, horizon = cfg.dual, cfg.guidance, cfg.horizon
    # Choose the model based on given cfg
    if guide==1:
        if dual==1:
            cfg.convgru['n_in'] = 6
            cfg.convgru['n_out'] = 2
            model_path = "train_size_100_guidance_1_dual_1_horizon_50_n_epochs_100.pth"
        else:
            cfg.convgru['n_in'] = 5
            cfg.convgru['n_out'] = 1
            model_path = "guidance_1_channel_2/train_size_100_guidance_1_horizon_50_n_epochs_100.pth"
    else:
        cfg.convgru['n_in'] = 1
        cfg.convgru['n_out'] = 1
        model_path = f"train_size_400_guidance_0_dual_0_channel_2_horizon_{horizon}_n_epochs_30.pth"  

    # Loading model
    model = StackUnstack(SimpleModel(**cfg.convgru)).to(default_device())
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        # Iterative forecasting
        val = get_data(ds).mean(axis=0, keepdims=True) 
        if dual==0:
            val = np.sum(val, axis=2, keepdims=True)
        val = np.log(val+1)    

        if guide:
            ip, lr, pmd, cam = [i for i in ds.split('x')[1:]]
            params = [int(ip), int(lr), float(pmd), float(cam)]
            val = add_guidance(val, params)

        val = torch.tensor(val[0]).float() 

        initial_seq = val[:horizon].to(default_device())
        inp = tuple(i.unsqueeze(0) for i in initial_seq)
        preds = [initial_seq[:,:dual+1,:,:].cpu().detach()]

        # Iteratively make prediction
        
        n_iter = N//horizon
        for iter in range(1, n_iter+1):
            p = model(inp)
            preds.append(torch.vstack(p).cpu().detach())

            if iter==n_iter:
                break

            if guide:
                target = val[iter*horizon:(iter+1)*horizon].to(default_device())
                guidance = target[:,dual+1:,:,:] 
                inp = tuple(torch.cat([p[i], guidance[i:i+1]], axis=1)  for i in range(horizon))

            else:
                inp = p
            
        preds = torch.cat(preds, axis=0)[:N]
        preds = torch.exp(preds)-1
        preds = preds.sum(axis=1, keepdims=True)

        if guide:
            val = val[:,:dual+1,:,:]
        val = torch.exp(val)-1
        val = val.sum(axis=1, keepdims=True)

    return val, preds




def get_pred_channel(ds, cfg, channel):
    """
    Do a foreacast for inactive (channel = 0) or active objects (channel = 1)
    """
    horizon = cfg.horizon

    cfg.convgru['n_in'] = 5
    cfg.convgru['n_out'] = 1
    if channel==1:
        model_path = "train_size_100_guidance_1_dual_0_channel_1_horizon_50_n_epochs_100.pth"

    else:
        model_path = "train_size_100_guidance_1_dual_0_channel_0_horizon_50_n_epochs_100.pth"

    model = StackUnstack(SimpleModel(**cfg.convgru)).to(default_device())
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        # Iterative forecasting
        path = f'{DATA_PATH}/TLE_density_all_{ds}.mat'
        mat = h5py.File(path, 'r')
        data = np.array(mat[KEY]).copy()
        data = data[:,:,:,:,:1] if channel == 0 else data[:,:,:,:,1:2]
        data = np.transpose(data, (0, 1, 4, 2, 3))

        val = data.mean(axis=0, keepdims=True) 
        val = np.log(val+1)    

        ip, lr, pmd, cam = [i for i in ds.split('x')[1:]]
        params = [int(ip), int(lr), float(pmd), float(cam)]
        val = add_guidance(val, params)
        val = torch.tensor(val[0]).float() 

        initial_seq = val[:horizon].to(default_device())
        inp = tuple(i.unsqueeze(0) for i in initial_seq)
        preds = [initial_seq[:,:1,:,:].cpu().detach()]
        
        n_iter = N//horizon
        for iter in range(1, n_iter+1):
            p = model(inp)
            preds.append(torch.vstack(p).cpu().detach())

            if iter==n_iter:
                break

            target = val[iter*horizon:(iter+1)*horizon].to(default_device())
            guidance = target[:,1:,:,:] 
            inp = tuple(torch.cat([p[i], guidance[i:i+1]], axis=1)  for i in range(horizon))

            
        preds = torch.cat(preds, axis=0)[:N]
        preds = torch.exp(preds)-1

        val = val[:,:1,:,:]
        val = torch.exp(val)-1

    return val, preds



def get_snapshots_combined(ds, cfg, log):
    """
    Make a comparison plot between models with different horizons over time 
    """
    horizons, years = [50], [5, 10, 20, 40, 70, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    snapshots = []
    for horizon in horizons:
        cfg.horizon = horizon
        val_0, preds_0 = get_pred_channel(ds, cfg, 0) # inactive objects
        val_1, preds_1 = get_pred_channel(ds, cfg, 1) # active objects
        val, preds = val_0 + val_1, preds_0 + preds_1 # combine them to get forecast of total number of objects
        snapshots.append(preds[indices].squeeze())

    val = get_data(ds).mean(axis=0) 
    val = np.sum(val, axis=1)

    if log==1:
        val = np.log(val+1)

    # Plot the predictions by horizon and year
    fig, axes = plt.subplots(len(horizons)+1, len(years), figsize=(16, 10))
    fig.suptitle("Predictions by Horizon and Year", fontsize=25)

    for i, preds_at_horizon in enumerate(snapshots):  # Each horizon's predictions
        for j, pred in enumerate(preds_at_horizon):  # Each year
            ax = axes[i, j]
            if log==1:
                pred = torch.log(pred +  1)
            ax.imshow(pred.numpy(), aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(f'Year {years[j]}', fontsize=FONT_SIZE)
            if j == 0:
                ax.set_ylabel(f'Horizon {horizons[i]}', fontsize=FONT_SIZE-3)


    for j, _ in enumerate(years):
        ax = axes[-1, j]
        ax.imshow(val[indices[j]], aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('Original Data', fontsize=FONT_SIZE-3)


    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    plt.savefig(f"td2/combined/{ds}.png", dpi=300)
    plt.show()    



def get_snapshots(ds, cfg, log):  
    """
    Make a comparison plot between models with different horizons over time 
    """
    horizons, years = [10, 25, 50], [5, 10, 20, 40, 70, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    snapshots = []
    for horizon in horizons:
        cfg.horizon = horizon
        val, preds = get_pred(cfg, ds)
        snapshots.append(preds[indices].squeeze())

    # Plotting and saving 
    val = get_data(ds).mean(axis=0) 
    val = np.sum(val, axis=1)

    if log==1:
        val = np.log(val+1)

    fig, axes = plt.subplots(len(horizons)+1, len(years), figsize=(16, 10))

    for i, preds_at_horizon in enumerate(snapshots):  
        for j, pred in enumerate(preds_at_horizon): 
            ax = axes[i, j]
            if log==1:
                pred = torch.log(pred +  1)
            ax.imshow(pred.numpy(), aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(f'Year {years[j]}', fontsize=FONT_SIZE)
            if j == 0:
                ax.set_ylabel(f'Horizon {horizons[i]}', fontsize=FONT_SIZE-3)

    for j, _ in enumerate(years):
        ax = axes[-1, j]
        ax.imshow(val[indices[j]], aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('Original Data', fontsize=FONT_SIZE-3)


    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    plt.savefig(f"td2/{ds}_guide_{cfg.guidance}.png", dpi=300)
    plt.show()

