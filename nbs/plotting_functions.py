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
from inference import get_data, get_pred, get_pred_channel, get_snapshots, get_snapshots_combined


DATA_PATH = "/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam" #TODO change it accordingly
N, KEY, LKB = 2436, 'comb_Am_rp', 50
NUMBER_SIZE, FONT_SIZE, TITLE_SIZE = 20, 20, 26



# Plotting SMAPEs over time
def smapes_over_time(cfg, ds_name):
    val, preds = get_pred(cfg, ds_name)
    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)

    new_smapes = [new(p, v).item() for p,v in zip(preds, val)]
    old_smapes = [old(p, v).item() for p,v in zip(preds, val)]
    total_smapes = [total(p, v).item() for p,v in zip(preds, val)]
    
    names = ['new', 'old', 'total']
    smape_lists = [new_smapes, old_smapes, total_smapes]

    for smapes, name in zip(smape_lists, names):
        _, ax = plt.subplots(figsize=(8, 6))  
        ax.plot(np.linspace(0, 100, len(smapes)), smapes, linewidth=2)
        ax.set_title(f"SMAPE Over Time", fontsize=FONT_SIZE)
        ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
        ax.set_ylabel("SMAPE", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
        plt.tight_layout()
        plt.savefig(f"evals/{name}_smape_{ds_name}.png") #change this accordingly
        plt.close()



def plot_smapes(cfg, ip, lr):
    """
    Plot SMAPEs across pmd and cam rates on a fixed initial population and a fixed launch rate
    """
    years = [10, 40, 50, 70, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    d = f'x{ip}x{lr}'
    pmds = [0.95, 0.96, 0.97, 0.98, 0.99]
    cams = [0.95, 0.96, 0.97, 0.98, 0.99]
    datasets = [d + f'x{pmd:.2f}x{cam:.2f}' for pmd in pmds for cam in cams]

    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)
    new_smapes, old_smapes, total_smapes = [], [], []

    for ds in tqdm(datasets):
        val, preds = get_pred(cfg, ds)
        val, preds = val[indices], preds[indices]

        new_smapes.append([new(p, v).item() for p,v in zip(preds, val)])
        old_smapes.append([old(p, v).item() for p,v in zip(preds, val)])
        total_smapes.append([total(p, v).item() for p,v in zip(preds, val)])


    new_smapes = np.array(new_smapes).reshape(len(cams), len(pmds), -1)
    old_smapes = np.array(old_smapes).reshape(len(cams), len(pmds), -1)
    total_smapes = np.array(total_smapes).reshape(len(cams), len(pmds), -1)
    

    for year_idx in range(len(years)):
        names = ['new', 'old', 'total']
        for smapes, name in zip([new_smapes, old_smapes, total_smapes], names):
            metrics = smapes[:, :, year_idx]

            # Create the heatmap
            plt.figure(figsize=(12, 12))
            heatmap = plt.imshow(metrics, cmap='Blues', aspect='auto', vmin=0, vmax=2, origin='lower')

            # Add colorbar
            cbar = plt.colorbar(heatmap)
            cbar.ax.tick_params(labelsize=20) 

            # Add labels, title, and ticks
            plt.title(f'Year {years[year_idx]}', fontsize=30)
            plt.xlabel('CAM', fontsize=27)
            plt.ylabel('PMD', fontsize=27)
            plt.xticks(np.arange(len(pmds)), pmds, fontsize=27)
            plt.yticks(np.arange(len(cams)), cams, fontsize=27)
            plt.tight_layout()

            # Change accordingly
            if cfg.guidance == 1:
                plt.savefig(f"td2/smapes/guidance/{name}_smape_{years[year_idx]}_x{ip}x{lr}.png")
            else:
                plt.savefig(f"td2/smapes/no_guidance/{name}_smape_{years[year_idx]}_x{ip}x{lr}.png")

            plt.show()
            plt.close()



def plot_smapes_combined(cfg, ip, lr):
    """
    Do forecast 
    """
    years = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    d = f'x{ip}x{lr}'
    pmds = [0.95, 0.96, 0.97, 0.98, 0.99]
    cams = [0.95, 0.96, 0.97, 0.98, 0.99]
    datasets = [d + f'x{pmd:.2f}x{cam:.2f}' for pmd in pmds for cam in cams]

    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)
    new_smapes, old_smapes, total_smapes = [], [], []

    for ds in tqdm(datasets):
        val_0, preds_0 = get_pred_channel(ds, cfg, 0)
        val_1, preds_1 = get_pred_channel(ds, cfg, 1)
        val, preds = val_0 + val_1, preds_0 + preds_1
        val, preds = val[indices], preds[indices]

        new_smapes.append([new(p, v).item() for p,v in zip(preds, val)])
        old_smapes.append([old(p, v).item() for p,v in zip(preds, val)])
        total_smapes.append([total(p, v).item() for p,v in zip(preds, val)])
    
    new_smapes = np.array(new_smapes).reshape(len(cams), len(pmds), -1)
    old_smapes = np.array(old_smapes).reshape(len(cams), len(pmds), -1)
    total_smapes = np.array(total_smapes).reshape(len(cams), len(pmds), -1)


    for year_idx in range(len(years)):
        names = ['new', 'old', 'total']
        smapes = [new_smapes]
        names = ['new']
        for smapes, name in zip([new_smapes, old_smapes, total_smapes], names):
            metrics = smapes[:, :, year_idx]

            # Create the heatmap
            plt.figure(figsize=(12, 12))
            heatmap = plt.imshow(metrics, cmap='Blues', aspect='auto', vmin=0, vmax=2, origin='lower')

            # Add colorbar
            cbar = plt.colorbar(heatmap)
            cbar.ax.tick_params(labelsize=20) 

            # Add labels, title, and ticks
            plt.title(f'Year {years[year_idx]}', fontsize=30)
            plt.xlabel('CAM', fontsize=27)
            plt.ylabel('PMD', fontsize=27)
            plt.xticks(np.arange(len(pmds)), pmds, fontsize=27)
            plt.yticks(np.arange(len(cams)), cams, fontsize=27)
            plt.tight_layout()
            plt.savefig(f"td2/smapes/channel/{name}_smape_{years[year_idx]}_x{ip}x{lr}_channel.png")
            plt.show()
            plt.close()



def plot_smapes_mixture(cfg, ip, lr, threshold):
    years = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    d = f'x{ip}x{lr}'
    pmds = [0.95, 0.96, 0.97, 0.98, 0.99]
    cams = [0.95, 0.96, 0.97, 0.98, 0.99]
    datasets = [d + f'x{pmd:.2f}x{cam:.2f}' for pmd in pmds for cam in cams]

    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)
    new_smapes, old_smapes, total_smapes = [], [], []

    for ds in tqdm(datasets):
        pmd = float(ds.split('x')[-2])
        if pmd <= threshold:
            val_0, preds_0 = get_pred_channel(ds, cfg, 0)
            val_1, preds_1 = get_pred_channel(ds, cfg, 1)
            val, preds = val_0 + val_1, preds_0 + preds_1
        else:
            cfg.guidance = 1
            cfg.dual = 0
            val, preds = get_pred(cfg, ds)

        val, preds = val[indices], preds[indices]

        new_smapes.append([new(p, v).item() for p,v in zip(preds, val)])
        old_smapes.append([old(p, v).item() for p,v in zip(preds, val)])
        total_smapes.append([total(p, v).item() for p,v in zip(preds, val)])


    new_smapes = np.array(new_smapes).reshape(len(cams), len(pmds), -1)
    old_smapes = np.array(old_smapes).reshape(len(cams), len(pmds), -1)
    total_smapes = np.array(total_smapes).reshape(len(cams), len(pmds), -1)
    

    os.makedirs(f"td2/smapes/mixture/x{ip}x{lr}")


    for year_idx in range(len(years)):
        names = ['new', 'old', 'total']
        name = ['new']

        for smapes, name in zip([new_smapes, old_smapes, total_smapes], names):
            metrics = smapes[:, :, year_idx]

            # Create the heatmap
            plt.figure(figsize=(12, 12))
            heatmap = plt.imshow(metrics, cmap='Blues', aspect='auto', vmin=0, vmax=2, origin='lower')

            # Add colorbar
            cbar = plt.colorbar(heatmap)
            cbar.ax.tick_params(labelsize=20) 

            # Add labels, title, and ticks
            plt.title(f'Year {years[year_idx]}', fontsize=30)
            plt.xlabel('CAM', fontsize=27)
            plt.ylabel('PMD', fontsize=27)
            plt.xticks(np.arange(len(pmds)), pmds, fontsize=27)
            plt.yticks(np.arange(len(cams)), cams, fontsize=27)
            plt.tight_layout()
            plt.savefig(f"td2/smapes/mixture/x{ip}x{lr}/{name}_smape_{years[year_idx]}_{threshold}.png")
            plt.show()
            plt.close()



def plot_smapes_with_without_guidance(ds_name, cfg):
    cfg.dual = 0
    cfg.guidance = 0
    val, preds = get_pred(cfg, ds_name)
    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)

    new_smapes = [new(p, v).item() for p,v in zip(preds, val)][::50]
    old_smapes = [old(p, v).item() for p,v in zip(preds, val)][::50]
    total_smapes = [total(p, v).item() for p,v in zip(preds, val)][::50]

    cfg.dual = 0
    cfg.guidance = 1
    val, preds = get_pred(cfg, ds_name)
    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)

    new_smapes_with_guidance = [new(p, v).item() for p,v in zip(preds, val)][::50]
    old_smapes_with_guidance = [old(p, v).item() for p,v in zip(preds, val)][::50]
    total_smapes_with_guidance = [total(p, v).item() for p,v in zip(preds, val)][::50]
    
    names = ['new', 'old', 'total']
    smape_lists = [new_smapes, old_smapes, total_smapes]
    smapes_guidance_list  = [new_smapes_with_guidance, old_smapes_with_guidance, total_smapes_with_guidance]

    for smapes, smapes_guidance, name in zip(smape_lists, smapes_guidance_list, names):
        _, ax = plt.subplots(figsize=(8, 6))  
        ax.plot(np.linspace(0, 100, len(smapes)), smapes, label = "Without guidance", linewidth=2)
        ax.plot(np.linspace(0, 100, len(smapes)), smapes_guidance, label = "With guidance", linewidth=2)

        ax.set_title(f"SMAPE Over Time", fontsize=FONT_SIZE)
        ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
        ax.set_ylabel("SMAPE", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
        ax.legend(fontsize=NUMBER_SIZE)
        plt.tight_layout()
        plt.savefig(f"zevals/{name}_smape_{ds_name}_comparison.png")
        plt.close()



def plot_losses_long_term(ds, cfg):

    all_losses = []
    for horizon in [10, 25, 50]:
        cfg.horizon = horizon
        val, preds = get_pred(cfg, ds)
        losses = [nn.L1Loss()(p,v).item() for p, v in zip(preds, val)]
        all_losses.append(losses)
    
    fig, ax = plt.subplots(figsize=(8, 6))  # One plot only

    for horizon, losses in zip([10, 25, 50], all_losses):
        losses = losses[::50]
        ax.plot(np.linspace(0, 100, len(losses)), losses, label=f'Horizon {horizon}')

    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("MAE loss", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"td2/{ds}_horizons_losses.png", dpi=300)
    plt.show()



def plot_recon_loss(ds, cfg):
    val, preds = get_pred(cfg, ds)
    losses = [nn.L1Loss()(p,v).item() for p, v in zip(preds, val)]
    
    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(np.linspace(0, 100, len(losses)), losses)

    ax.set_title(f"Forecasting error over time", fontsize=FONT_SIZE)
    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("MAE loss", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    plt.tight_layout()
    plt.savefig(f"zevals/{ds}_forecast_loss.png", dpi=300)
    plt.show()



def plot_loss_with_without_guidance(ds, cfg):
    
    cfg.guidance = 0
    cfg.convgru['n_in'] = 1

    val, preds = get_pred(cfg, ds)
    losses_no_guidance = [nn.L1Loss()(p,v).item() for p, v in zip(preds, val)]

    cfg.guidance = 1
    cfg.convgru['n_in'] = 5

    val, preds = get_pred(cfg, ds)
    losses_with_guidance = [nn.L1Loss()(p,v).item() for p, v in zip(preds, val)]
    
    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(np.linspace(0, 100, N), losses_no_guidance, label="Without guidance")
    ax.plot(np.linspace(0, 100, N), losses_with_guidance, label="With guidance")

    ax.set_title(f"Forecasting error over time", fontsize=FONT_SIZE)
    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("MAE loss", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    plt.tight_layout()
    plt.savefig(f"zevals/{ds}_guidance_comparison_loss.png", dpi=300)
    plt.show()



def plot_number_of_objects_with_without_guidance(ds, cfg):

    cfg.dual = 0
    cfg.guidance = 0
    val, preds = get_pred(cfg, ds)
    number_of_objects_no_guidance = preds.sum(axis=(-1,-2)).squeeze()[::50]

    cfg.guidance = 1
    val, preds = get_pred(cfg, ds)
    number_of_objects_with_guidance = preds.sum(axis=(-1,-2)).squeeze()[::50]

    MC = val.sum(axis=(-1,-2)).squeeze()[::50]

    n = len(MC)

    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(np.linspace(0, 100, n), MC, label="Target")
    ax.plot(np.linspace(0, 100, n), number_of_objects_no_guidance, label="Without guidance")
    ax.plot(np.linspace(0, 100, n), number_of_objects_with_guidance, label="With guidance")

    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("Number of objects", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)

    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"td2/{ds}_comparison_number_of_objects.png", dpi=300)
    plt.show()



def plot_number_of_objects_with_without_guidance_dual(ds, cfg):

    cfg.guidance = 1
    cfg.dual = 1
    val, preds = get_pred(cfg, ds)
    number_of_objects_dual = preds.sum(axis=(-1,-2)).squeeze()[::50]

    cfg.dual = 0
    val, preds = get_pred(cfg, ds)
    number_of_objects_no_dual = preds.sum(axis=(-1,-2)).squeeze()[::50]

    MC = val.sum(axis=(-1,-2)).squeeze()[::50]

    n = len(MC)

    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(np.linspace(0, 100, n), MC, label="Target")
    ax.plot(np.linspace(0, 100, n), number_of_objects_dual, label="Dual")
    ax.plot(np.linspace(0, 100, n), number_of_objects_no_dual, label="No Dual")

    ax.set_title(f"Number of objects over time", fontsize=FONT_SIZE)
    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("Number of objects", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.tight_layout()
    plt.savefig(f"zevals/{ds}_guidance_comparison_number_of_objects_dual.png", dpi=300)
    plt.show()



def plot_number_of_objects(ds, cfg):

    val, preds = get_pred(cfg, ds)
    val, preds = val.squeeze(), preds.squeeze()
    val_objects, preds_objects = val.sum(axis=(-1,-2)), preds.sum(axis=(-1,-2))

    date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())
    os.makedirs(f'zevals/{date}')

    _, ax = plt.subplots(figsize=(8, 6))  #
    ax.plot(np.linspace(0, 100, N), val_objects, label='Target')
    ax.plot(np.linspace(0, 100, N), preds_objects, label='Prediction')

    ax.set_title(f"Total Number of Objects Over Time", fontsize=FONT_SIZE)
    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("Number of objects", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.tight_layout()
    plt.savefig(f"zevals/{date}/{ds}_total_number_objects_{cfg.guidance}.png", dpi=300)
    plt.show()



def plot_num_objects_channel(ds, cfg, channel):
    val, preds = get_pred_channel(ds, cfg, channel)
    val, preds = val.squeeze(), preds.squeeze()
    val_objects, preds_objects = val.sum(axis=(-1,-2)), preds.sum(axis=(-1,-2))

    _, ax = plt.subplots(figsize=(8, 6))  # One plot only

    ax.plot(np.linspace(0, 100, N), val_objects, label='Target')
    ax.plot(np.linspace(0, 100, N), preds_objects, label='Prediction')

    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("Number of objects", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"td2/{ds}_total_number_objects_{cfg.guidance}_channel_{channel}.png", dpi=300)
    plt.show()



if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Model evaluation")

    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--guidance", type = int, default = 0)  
    parser.add_argument("--dual", type = int, default = 0)  

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)

    # Settings
    model_type = 'convgru'
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)

    config = AttrDict(config_base)

    keys = ['horizon',  'guidance', 'dual']  
    for key in keys:
        config[key] = arg_dict[key]

    config['log'] = 1
    
    ds = "x2x2x0.99x0.99"

    plot_num_objects_channel(ds, config, 1)

